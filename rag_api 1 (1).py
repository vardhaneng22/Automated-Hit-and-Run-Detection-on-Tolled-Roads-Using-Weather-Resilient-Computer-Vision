import os
import cv2
import time
import re
import numpy as np
import easyocr
from ultralytics import YOLO

# ======================================
# Load Models
# ======================================
print("Loading YOLO car model (auto-download first time)...")
yolo_model = YOLO("yolov8n.pt")   # official model
reader = easyocr.Reader(['en'], gpu=False)


# ======================================
# Image Enhancement (Retinex + CLAHE)
# ======================================
def enhance_image(image):
    img = image.astype(np.float32) + 1.0
    b, g, r = cv2.split(img)

    def retinex(channel):
        blur = cv2.GaussianBlur(channel, (0, 0), 30)
        return np.log(channel) - np.log(blur + 1)

    r = retinex(r)
    g = retinex(g)
    b = retinex(b)

    merged = cv2.merge((b, g, r))
    merged = cv2.normalize(merged, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lab = cv2.cvtColor(merged, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    l = clahe.apply(l)
    final = cv2.merge((l, a, b))
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    return final


# ======================================
# Detect Car using YOLO
# ======================================
def detect_car(image):
    results = yolo_model(image)[0]

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 2:  # class 2 = car
            x1, y1, x2, y2 = map(int, box)
            print("🚗 Car detected")
            return image[y1:y2, x1:x2]

    print("⚠ No car detected, using full image")
    return image


# ======================================
# Detect Plate using Contours
# ======================================
def detect_plate_from_car(car_img):
    gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h)

        if 2 < ratio < 5 and w > 80 and h > 25:
            print("🔍 Plate detected via contour")
            return car_img[y:y+h, x:x+w]

    # fallback region
    print("⚠ Using fallback plate region")
    h, w = car_img.shape[:2]
    return car_img[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]


# ======================================
# Deskew Plate
# ======================================
def deskew_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 50:
        return plate

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle

    (h, w) = plate.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        plate, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    print(f"🔄 Deskew angle: {angle:.2f}")
    return rotated


# ======================================
# Prepare Plate for OCR (No hardcoding fix)
# ======================================
def prepare_for_ocr(plate):
    # Add border (important)
    plate = cv2.copyMakeBorder(
        plate, 20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Strong upscale
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    return thresh

# ======================================
# OCR (confidence-based, allowlist)
# ======================================
def recognize_text(img):
    texts = []

    # Mode 1: Original (minimal processing)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res1 = reader.readtext(
        rgb,
        detail=1,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    texts += res1

    # Mode 2: Grayscale + resize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    res2 = reader.readtext(
        gray,
        detail=1,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    texts += res2

    # Mode 3: Threshold (for dark/noisy plates)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res3 = reader.readtext(
        thresh,
        detail=1,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    texts += res3

    if not texts:
        return ""

    # Combine results left-to-right
    texts = sorted(texts, key=lambda x: x[0][0][0])

    # Choose result with maximum characters (most complete)
    best = max(texts, key=lambda x: len(x[1]))

    text = best[1].upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


# ======================================
# Indian Plate Format Validation
# ======================================
def clean_text(text):
    pattern = r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}"
    match = re.search(pattern, text)
    return match.group() if match else text


# ======================================
# MAIN PIPELINE
# ======================================
def main(image_path):
    start = time.perf_counter()

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Image not found")
        return

    # Enhance full image
    image = enhance_image(image)
    print("☀ Image enhanced")

    # Detect car
    car = detect_car(image)

    # Detect plate
    plate = detect_plate_from_car(car)

    if plate is None or plate.size == 0:
        print("❌ Plate not detected")
        return

    # Deskew
    plate = deskew_plate(plate)

    # Save debug
    os.makedirs("outputs/debug", exist_ok=True)
    cv2.imwrite("outputs/debug/plate.jpg", plate)
    print("📸 Plate saved at outputs/debug/plate.jpg")

    # OCR
    raw_text = recognize_text(plate)
    print("RAW OCR:", raw_text)

    final_text = clean_text(raw_text)
    print("✅ Final Plate:", final_text)

    end = time.perf_counter()
    print(f"⏱ Runtime: {end - start:.2f} sec")


# ======================================
# Run
# ======================================
if __name__ == "__main__":
    main("C:\\Users\\Priyanka\\Desktop\\RAG_Pipeline\\image1.png")