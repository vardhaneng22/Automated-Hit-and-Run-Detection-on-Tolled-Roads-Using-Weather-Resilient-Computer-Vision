import os
import cv2
import time
import re
import numpy as np
import easyocr
from ultralytics import YOLO

print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

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

def detect_car(image):
    results = yolo_model(image)[0]
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) == 2:
            x1, y1, x2, y2 = map(int, box)
            return image[y1:y2, x1:x2], (x1, y1)
    return None, None

def detect_plate_from_car(car_img):
    gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # Enhance contrast and use adaptive threshold to reveal plate-like rectangles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 0
        area = w * h
        if 2.0 <= ratio <= 6.0 and area > 1500 and w > 60 and h > 15:
            candidates.append((x, y, w, h))

    if not candidates:
        return None, None

    # score candidates by character-like edge density and aspect ratio closeness
    best_score = -1
    best_bbox = None
    for x, y, w, h in candidates:
        roi_gray = gray[y:y+h, x:x+w]
        edges_roi = cv2.Canny(roi_gray, 50, 150)
        edge_count = int(np.count_nonzero(edges_roi))
        area = float(max(1, w * h))
        edge_density = edge_count / area
        ratio = w / float(h) if h > 0 else 0
        ratio_score = 1.0 - (abs(ratio - 4.0) / 4.0)
        score = edge_density * 0.8 + max(ratio_score, 0.0) * 0.2
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    x, y, w, h = best_bbox
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(car_img.shape[1], x + w + pad_x)
    y2 = min(car_img.shape[0], y + h + pad_y)
    return car_img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def detect_plate_fixed(image):
    h, w = image.shape[:2]
    y1, y2 = int(h*0.55), int(h*0.80)
    x1, x2 = int(w*0.20), int(w*0.80)
    return image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

def detect_plate_fallback(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    max_area = 0

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        ratio = w / float(h)
        area = w*h
        if 2 < ratio < 6 and 3000 < area < 80000:
            if area > max_area:
                best = (x,y,w,h)
                max_area = area

    if best:
        x,y,w,h = best
        return image[y:y+h, x:x+w], best

    return None, None


def detect_plate_mser(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(_delta=5)
    regions, _ = mser.detectRegions(gray)
    boxes = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        ratio = w / float(h) if h > 0 else 0
        area = w * h
        if 2.0 <= ratio <= 8.0 and area > 500:
            boxes.append((x, y, w, h))

    if not boxes:
        return None, None

    # score by edge density similar to earlier
    best_score = -1
    best = None
    for x, y, w, h in boxes:
        rx = max(0, x)
        ry = max(0, y)
        rw = min(image.shape[1] - rx, w)
        rh = min(image.shape[0] - ry, h)
        roi = gray[ry:ry+rh, rx:rx+rw]
        if roi.size == 0:
            continue
        edges = cv2.Canny(roi, 50, 150)
        edge_count = int(np.count_nonzero(edges))
        area = float(max(1, rw * rh))
        edge_density = edge_count / area
        ratio = rw / float(rh) if rh > 0 else 0
        ratio_score = 1.0 - (abs(ratio - 4.0) / 4.0)
        score = edge_density * 0.8 + max(ratio_score, 0.0) * 0.2
        if score > best_score:
            best_score = score
            best = (rx, ry, rw, rh)

    if best is None:
        return None, None
    x, y, w, h = best
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.12)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def validate_plate_format(text):
    # Prefer Indian format: AA00AA0000 (approx) e.g. KA01AB1234
    indian = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$')
    if indian.match(text):
        return True
    # General fallback: must contain letters and digits and length between 6 and 12
    if 6 <= len(text) <= 12 and re.search(r'[A-Z]', text) and re.search(r'[0-9]', text):
        return True
    return False


def recognize_text(plate, top_k=3):
    """Return best plate text plus candidates list (text, confidence).
    Uses EasyOCR detail=1 to collect text with confidence and scores candidates
    based on format validity and confidence."""
    if plate is None or plate.size == 0:
        return "", []

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    raw = reader.readtext(gray, detail=1, paragraph=False)
    candidates = []
    for item in raw:
        # item: (bbox, text, conf)
        if len(item) == 3:
            txt = re.sub(r'[^A-Z0-9]', '', item[1].upper())
            conf = float(item[2])
            if txt:
                score = conf
                if validate_plate_format(txt):
                    score += 30.0
                # prefer mixed alnum
                if re.search(r'[A-Z]', txt) and re.search(r'[0-9]', txt):
                    score += 5.0
                candidates.append((score, txt, conf))

    # dedupe by text keeping highest score
    best_map = {}
    for score, txt, conf in candidates:
        if txt not in best_map or score > best_map[txt][0]:
            best_map[txt] = (score, conf)

    final_candidates = sorted([(v[0], k, v[1]) for k, v in best_map.items()], reverse=True)
    # keep top_k
    final_candidates = final_candidates[:top_k]
    best_text = final_candidates[0][1] if final_candidates else ""
    return best_text, [(c[1], c[2]) for c in final_candidates]

def clean_plate(text):
    m = re.search(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}", text)
    return m.group() if m else text

def main(image_path):
    start = time.time()

    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return

    image = enhance_image(image)

    plate = None
    box = None

    car, offset = detect_car(image)
    if car is not None:
        plate, b = detect_plate_from_car(car)
        if plate is not None:
            x,y,w,h = b
            ox, oy = offset
            box = (x+ox, y+oy, w, h)

    if plate is None:
        # If we have a car crop, try a fixed region on the car (better than full-image fixed box)
        if car is not None:
            plate, b = detect_plate_fixed(car)
            if plate is not None:
                x,y,w,h = b
                ox, oy = offset
                box = (x+ox, y+oy, w, h)
        # Fallback to a fixed region on full image if still None
        if plate is None:
            plate, box = detect_plate_fixed(image)

    best_text, candidates = recognize_text(plate)

    # Prefer candidates that match plate format; otherwise try fallback
    def pick_valid(cands):
        for txt, conf in cands:
            if validate_plate_format(txt):
                return txt
        for txt, conf in cands:
            if 6 <= len(txt) <= 12 and re.search(r'[A-Z]', txt) and re.search(r'[0-9]', txt):
                return txt
        return None

    picked = pick_valid(candidates)
    if picked:
        best_text = picked
    else:
        best_text = ""

    if best_text == "":
        plate, box = detect_plate_fallback(image)
        if plate is not None:
            best_text, candidates = recognize_text(plate)
            picked = pick_valid(candidates)
            if picked:
                best_text = picked

    # Try MSER-based candidate search if still not found
    if best_text == "":
        plate_m, box_m = detect_plate_mser(image)
        if plate_m is not None:
            best_m, cands_m = recognize_text(plate_m)
            picked_m = pick_valid(cands_m)
            if picked_m:
                plate = plate_m
                box = box_m
                best_text = picked_m

    if plate is None:
        print("Plate not detected")
        return

    final = clean_plate(best_text)

    print("RAW OCR candidates:", candidates)
    print("FINAL PLATE:", final)

    x,y,w,h = box
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.putText(image, final, (x,y+h+30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/result.jpg", image)
    cv2.imwrite("outputs/plate_crop.jpg", plate)
    # Avoid imshow crashing in headless builds
    try:
        cv2.imshow("Number Plate Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("Saved outputs/result.jpg")
    print("Runtime:", round(time.time()-start,2), "sec")

if __name__ == "__main__":
    main("i.jpg")