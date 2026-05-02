import os
import cv2
import time
import re
import numpy as np
import easyocr
import sys
from ultralytics import YOLO

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def normalize_lighting(image):

    img = image.astype(np.float32) + 1.0
    b, g, r = cv2.split(img)

    def retinex(channel):
        blur = cv2.GaussianBlur(channel, (0, 0), sigmaX=30)
        ret = np.log(channel) - np.log(blur + 1)
        return ret

    r_ret = retinex(r)
    g_ret = retinex(g)
    b_ret = retinex(b)

    retinex_img = cv2.merge((b_ret, g_ret, r_ret))

    retinex_img = cv2.normalize(
        retinex_img, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    lab = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    final = cv2.merge((l, a, b))
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    return final

def night_vision_enhance(image):

    img = image.astype(np.float32) / 255.0
    illumination = cv2.GaussianBlur(img, (0, 0), sigmaX=15)
    reflectance = img / (illumination + 1e-6)

    reflectance = cv2.normalize(
        reflectance, None, 0, 1, cv2.NORM_MINMAX
    )

    enhanced = (reflectance * 255).astype(np.uint8)

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    gamma = 1.8
    final = np.power(final / 255.0, 1 / gamma)
    final = (final * 255).astype(np.uint8)

    return final

def detect_plate_roi(image):
    h, w = image.shape[:2]

    y1 = int(h * 0.55)
    y2 = int(h * 0.80)
    x1 = int(w * 0.20)
    x2 = int(w * 0.80)

    fallback_plate = image[y1:y2, x1:x2]
    return fallback_plate

yolo_model = YOLO("models/plate_detector/yolov8n.pt")

def detect_plate_yolo(image):
    results = yolo_model(image, conf=0.4)

    if len(results[0].boxes) == 0:
        return None, None

    box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    plate_img = image[y1:y2, x1:x2]

    return plate_img, box

reader = easyocr.Reader(['en'], gpu=False)

def recognize_text(plate_img):
    if len(plate_img.shape) == 2:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2RGB)
    else:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

    results = reader.readtext(
        plate_img,
        detail=0,
        paragraph=False
    )

    if not results:
        return ""

    text = "".join(results)
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    return text

def clean_plate_text(text):
    pattern = r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}"
    match = re.search(pattern, text)
    return match.group() if match else text


def format_plate_with_spaces(text):
    text = text.replace(" ", "")
    match = re.match(r"([A-Z]{2})([0-9]{1,4})([A-Z]{1,2})", text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return text


def format_indian_plate(text):
    text = text.replace(" ", "").upper()

    match_full = re.match(r"([A-Z]{2})([0-9]{1,2})([A-Z]{1,2})([0-9]{3,4})", text)
    if match_full:
        return f"{match_full.group(1)} {match_full.group(2)} {match_full.group(3)} {match_full.group(4)}"

    match_short = re.match(r"([A-Z]{2})([0-9]{1,4})([A-Z]{1,2})", text)
    if match_short:
        return f"{match_short.group(1)} {match_short.group(2)} {match_short.group(3)}"

    return text

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception("Image not found")
    return img


def detect_plate_robust(image, car_model):
    """Detect vehicles first, then localize plate candidates within them."""
    h_img, w_img = image.shape[:2]
    results = car_model(image, conf=0.3, verbose=False)[0]
    
    # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
    vehicle_classes = {2, 3, 5, 7}
    best_candidate = None
    best_score = 0

    # Collect all vehicle crops
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) in vehicle_classes:
            x1, y1, x2, y2 = map(int, box)
            # Add small padding
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(w_img, x2 + 10), min(h_img, y2 + 10)
            
            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.size == 0: continue

            # Focus on lower half of the vehicle for plates
            vh, vw = vehicle_crop.shape[:2]
            roi_y1 = int(vh * 0.4)
            roi = vehicle_crop[roi_y1:vh, :]
            
            # Morphology to find rectangles
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            # Find vertical edges (plates have high contrast text)
            edged = cv2.Canny(gray, 30, 200)
            
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                # We are looking for a quadrilateral
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Typical Indian plate is ~ 4:1 to 5:1
                if 2.0 <= aspect_ratio <= 6.0 and w > 40 and h > 10:
                    area = w * h
                    if area > best_score:
                        best_score = area
                        # Crop from full image for better resolution
                        cx1, cy1 = x1 + x, y1 + roi_y1 + y
                        cx2, cy2 = x1 + x + w, y1 + roi_y1 + y + h
                        # Extra padding for OCR
                        px, py = int(w * 0.05), int(h * 0.1)
                        cx1, cy1 = max(0, cx1 - px), max(0, cy1 - py)
                        cx2, cy2 = min(w_img, cx2 + px), min(h_img, cy2 + py)
                        best_candidate = image[cy1:cy2, cx1:cx2]

    # Fallback to YOLO box if any 
    if best_candidate is None:
        # Check if there's any box detected by the current model
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            if conf > 0.4:
                x1, y1, x2, y2 = map(int, box)
                return image[y1:y2, x1:x2]
        
        # EFFECTIVE FALLBACK (From previous success)
        y1, y2 = int(h_img * 0.55), int(h_img * 0.80)
        x1, x2 = int(w_img * 0.20), int(w_img * 0.80)
        print(f"  Fallback ROI applied: {x1, y1, x2, y2}")
        return image[y1:y2, x1:x2]

    return best_candidate

def detect_plate(image):
    # This wrapper maintains compatibility with existing main
    return detect_plate_robust(image, yolo_model)


def main(image_path):
    start_time = time.perf_counter()
    
    filename = os.path.basename(image_path).split('.')[0]
    image = load_image(image_path)

    _ = preprocess_image(image)

    try:
        image = normalize_lighting(image)
        print("☀️ Light normalized")
    except Exception as e:
        print("Light normalization skipped:", e)

    try:
        image = night_vision_enhance(image)
        print("🌙 Night vision applied")
    except Exception as e:
        print("Night vision skipped:", e)

    plate_img = detect_plate(image)
    if plate_img is None:
        print("No plate detected")
        return

    os.makedirs("outputs/debug", exist_ok=True)
    out_crop_path = f"outputs/debug/{filename}_test_plate.jpg"
    cv2.imwrite(out_crop_path, plate_img)
    print(f"Saved crop to {out_crop_path}")

    raw_text = recognize_text(plate_img)
    print("RAW OCR:", raw_text)

    final_text = clean_plate_text(raw_text)
    print("Final:", final_text)

    print("Formatted:", format_indian_plate(final_text))

    end_time = time.perf_counter()
    print(f"Runtime: {end_time - start_time:.3f} sec")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("i2.jpg")