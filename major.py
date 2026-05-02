"""
major.py  -  Car Damage Detection  +  License Plate Recognition (ANPR)
======================================================================
Handles: natural light, dim light, low light / night images.
Handles: small / low-resolution input images via smart upscaling.

Usage:
    python major.py                   # auto-runs on d1.jpg
    python major.py <image_path>      # runs on a single image

Two separate matplotlib windows per image:
    Window 1  ->  Damage Detection   (Original | Enhanced | YOLO Detections)
    Window 2  ->  Plate Recognition  (Enhanced | Plate Crop | OCR Result)
"""

import os, sys, re, time, warnings
import requests
import cv2
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import easyocr
from ultralytics import YOLO

# optional Tesseract fallback for tough plates
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

warnings.filterwarnings("ignore")

# ===================================================================
#                         CONFIGURATION
# ===================================================================
DAMAGE_WEIGHTS_URL  = "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector/raw/main/trained.pt"
DAMAGE_WEIGHTS_FILE = "trained.pt"
GENERAL_YOLO_FILE   = "yolov8n.pt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Extra output directory for debugging
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimum dimensions for reliable YOLO detection
MIN_DIM_FOR_YOLO = 416  # reduced from 640 for faster processing

# ===================================================================
#                       MODEL LOADING
# ===================================================================
def _download_weights(url, filename):
    fp = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(fp):
        return
    print(f"Downloading {filename} ...")
    r = requests.get(url, stream=True); r.raise_for_status()
    with open(fp, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print("Download complete.")

def load_models():
    _download_weights(DAMAGE_WEIGHTS_URL, DAMAGE_WEIGHTS_FILE)
    print("Loading YOLO Damage Detector ...")
    damage_model = YOLO(os.path.join(SCRIPT_DIR, DAMAGE_WEIGHTS_FILE))
    print("Loading YOLOv8n (car detection) ...")
    car_model = YOLO(GENERAL_YOLO_FILE)
    print("Loading EasyOCR ...")
    reader = easyocr.Reader(["en"], gpu=False)
    return damage_model, car_model, reader


# ===================================================================
#                   SMART UPSCALING
# ===================================================================
def smart_upscale(image, target_min_dim=MIN_DIM_FOR_YOLO):
    """
    Upscale small images so the shortest side is at least target_min_dim.
    Uses INTER_CUBIC for quality. Returns (upscaled_image, scale_factor).
    """
    h, w = image.shape[:2]
    short_side = min(h, w)

    if short_side >= target_min_dim:
        return image.copy(), 1.0

    scale = target_min_dim / short_side
    # Cap scale to avoid extreme enlargement
    scale = min(scale, 4.0)

    new_w = int(w * scale)
    new_h = int(h * scale)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return upscaled, scale


# ===================================================================
#            ADAPTIVE IMAGE ENHANCEMENT  (works all lighting)
# ===================================================================
def assess_brightness(image):
    """Return mean brightness 0-255 of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def detect_shadow_regions(image):
    """
    Detect dark/shadow regions in the image.
    Returns a binary mask where 255=shadow, 0=non-shadow.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Values below 80 are considered shadows
    _, shadow_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    # Dilate shadow mask to expand shadow regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)
    return shadow_mask

def is_in_shadow(box, shadow_mask):
    """
    Check if a detection box overlaps significantly with shadow region.
    Returns True if >40% of box is in shadow.
    """
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(shadow_mask.shape[1], x2), min(shadow_mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    roi = shadow_mask[y1:y2, x1:x2]
    shadow_ratio = np.sum(roi > 0) / (roi.size + 1e-6)
    return shadow_ratio > 0.4

def enhance_for_damage(image):
    """
    Contrast enhancement for damage detection.
    Histogram stretch + CLAHE. Adaptive to lighting.
    """
    brightness = assess_brightness(image)
    img_f = image.astype(np.float32)

    # Gamma correction for dark images
    if brightness < 100:
        gamma = 0.5 if brightness < 50 else 0.7
        table = np.array([((i / 255.0) ** (1.0/gamma)) * 255
                          for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)
        img_f = image.astype(np.float32)

    # Histogram stretch (de-fog)
    lo = np.percentile(img_f, 1)
    hi = np.percentile(img_f, 99)
    if hi - lo > 0:
        stretched = (img_f - lo) * (255.0 / (hi - lo))
    else:
        stretched = img_f
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    # Sharpen
    blur = cv2.GaussianBlur(stretched, (0, 0), 3.0)
    sharpened = cv2.addWeighted(stretched, 1.5, blur, -0.5, 0)

    # CLAHE - stronger for dark images
    clip_limit = 4.0 if brightness < 80 else 2.0
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return result

def enhance_for_plate(image):
    """
    Enhancement specifically for plate reading.
    Uses aggressive CLAHE + denoising for low-light.
    """
    brightness = assess_brightness(image)

    # For very dark images, apply gamma correction first
    if brightness < 100:
        gamma = 0.4 if brightness < 50 else 0.6
        table = np.array([((i / 255.0) ** (1.0/gamma)) * 255
                          for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # CLAHE on L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Sharpen edges
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    return enhanced


# ===================================================================
#       PIPELINE 1:  DAMAGE DETECTION
# ===================================================================
def run_damage_detection(image_path, original, damage_model, fig_num):
    # Upscale for better YOLO detection
    upscaled, scale = smart_upscale(original)
    enhanced = enhance_for_damage(upscaled.copy())

    # Detect shadow regions to filter false positives
    shadow_mask = detect_shadow_regions(enhanced)

    # Use lower confidence for dark images
    brightness = assess_brightness(original)
    conf_thresh = 0.25 if brightness < 100 else 0.35

    # YOLO predict with explicit image size
    results = damage_model.predict(enhanced, save=False, conf=conf_thresh,
                                    imgsz=max(enhanced.shape[:2]))

    for result in results:
        # Filter out detections in shadow regions (false positives)
        filtered_boxes = []
        filtered_confs = []
        filtered_classes = []
        
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf,
                                   result.boxes.cls):
            # Skip if detection is mostly in shadow
            if not is_in_shadow(box, shadow_mask):
                filtered_boxes.append(box)
                filtered_confs.append(conf)
                filtered_classes.append(cls)
        
        # Replace with filtered detections
        if filtered_boxes:
            result.boxes.xyxy = np.array(filtered_boxes)
            result.boxes.conf = np.array(filtered_confs)
            result.boxes.cls = np.array(filtered_classes)
        else:
            # If all filtered out, keep original but with higher threshold
            pass
        
        res_plotted = result.plot()

        orig_rgb  = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        enh_rgb   = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        res_rgb   = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(22, 7), num=fig_num)
        fig.suptitle(f"DAMAGE DETECTION  -  {os.path.basename(image_path)}",
                     fontsize=14, fontweight="bold")
        for ax, img, title in zip(axes,
            [orig_rgb, enh_rgb, res_rgb],
            ["Original", "Enhanced", "YOLO Damage Detection"]):
            ax.imshow(img); ax.set_title(title, fontsize=12); ax.axis("off")
        plt.tight_layout()
        
        # Save figure to output directory
        fig_path = os.path.join(OUTPUT_DIR, f"damage_{os.path.basename(image_path)}")
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        print(f"  Damage figure saved to: {fig_path}")

        # Console report
        print(f"\n{'='*50}")
        print(f"  DAMAGE REPORT  -  {os.path.basename(image_path)}")
        print(f"{'='*50}")
        if len(result.boxes) == 0:
            print("  No damage detected.")
        else:
            for box in result.boxes:
                cls_name = damage_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"  * {cls_name.upper()}  (confidence {conf:.0%})")
        return fig


# ===================================================================
#       PIPELINE 2:  LICENSE PLATE RECOGNITION (ANPR)
# ===================================================================

# ---------- Step A: Detect car region ----------
def detect_car_region(image, car_model):
    results = car_model(image, verbose=False)[0]
    best_box = None
    best_area = 0
    # Accept car(2), truck(7), bus(5) for wider detection
    vehicle_classes = {2, 5, 7}
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls) in vehicle_classes:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
    if best_box:
        x1, y1, x2, y2 = best_box
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        print("  Car/vehicle detected")
        return image[y1:y2, x1:x2]
    print("  No vehicle bbox - using full image")
    return image


# ---------- Step B: Find plate candidates (multi-strategy) ----------
def find_plate_candidates(car_img):
    """
    Multi-strategy plate localization.
    Returns list of (crop, score, method) sorted by score descending.
    """
    candidates = []
    h_img, w_img = car_img.shape[:2]
    # Very low minimums for small images
    min_plate_w = max(30, w_img * 0.05)
    min_plate_h = max(8, h_img * 0.02)

    # grayscale + equalized copy (helps in low-light situations)
    gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # --- Strategy 0: Haar cascade (built-in OpenCV) ---
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_russian_plate_number.xml")
    plates = cascade.detectMultiScale(gray_eq, scaleFactor=1.15,
                                      minNeighbors=3,
                                      minSize=(int(min_plate_w),
                                               int(min_plate_h)))
    for (x, y, w, h) in plates:
        # AGGRESSIVE padding to capture FULL plate without cutoff
        px, py = int(w * 0.2), int(h * 0.35)
        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(w_img, x + w + px)
        y2 = min(h_img, y + h + py)
        crop = car_img[y1:y2, x1:x2]
        ratio_bonus = max(0, 1.0 - abs((w/h) - 4.5) / 4.5)
        candidates.append((crop, w * h * 1.2 + ratio_bonus * 1000, "haar"))

    # --- Strategy 1: Fast morphological (single kernel) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dilate to connect broken characters and expand region
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_thresh = cv2.dilate(thresh, dilate_k, iterations=2)
    
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(dilated_thresh, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours[:20]:  # limit to top 20 contours
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 0
        if 1.5 < ratio < 7.0 and w > min_plate_w and h > min_plate_h:
            # AGGRESSIVE padding to capture FULL plate
            px, py = int(w * 0.2), int(h * 0.35)
            x1 = max(0, x - px)
            y1 = max(0, y - py)
            x2 = min(w_img, x + w + px)
            y2 = min(h_img, y + h + py)
            crop = car_img[y1:y2, x1:x2]
            ratio_bonus = max(0, 1.0 - abs(ratio - 4.5) / 4.5)
            score = (w * h) * 0.5 + (y / h_img) * 100 + ratio_bonus * 300
            candidates.append((crop, score, "morph"))

    # --- Strategy 2: Edge detection with better post-processing ---
    blur = cv2.bilateralFilter(gray_eq, 9, 15, 15)  # lighter filter
    edges = cv2.Canny(blur, 50, 150)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilated = cv2.dilate(edges, dilate_k, iterations=2)  # more dilation

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 0
        if 1.5 < ratio < 7.0 and w > min_plate_w and h > min_plate_h:
            # AGGRESSIVE padding to capture FULL plate without cutoff
            px, py = int(w * 0.2), int(h * 0.35)
            x1, y1 = max(0, x - px), max(0, y - py)
            x2, y2 = min(w_img, x + w + px), min(h_img, y + h + py)
            crop = car_img[y1:y2, x1:x2]
            score = (w * h) * 0.3 + (y / h_img) * 80
            candidates.append((crop, score, "edge"))

    # --- Strategy 3: Fallback regions ---
    regions = [
        # (y_start_frac, y_end_frac, x_start_frac, x_end_frac)
        (0.60, 0.95, 0.05, 0.95),   # bottom most of car
        (0.50, 0.80, 0.10, 0.90),   # lower-mid
        (0.00, 0.35, 0.10, 0.90),   # top (front plate)
        (0.70, 1.00, 0.00, 1.00),   # very bottom full width
        (0.00, 0.25, 0.00, 1.00),   # very top full width
    ]
    for (fy1, fy2, fx1, fx2) in regions:
        r_y1 = max(0, int(h_img * fy1))
        r_y2 = min(h_img, int(h_img * fy2))
        r_x1 = max(0, int(w_img * fx1))
        r_x2 = min(w_img, int(w_img * fx2))
        if r_y2 > r_y1 and r_x2 > r_x1:
            crop = car_img[r_y1:r_y2, r_x1:r_x2]
            candidates.append((crop, 10, "fallback_region"))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


# ---------- Step C: Prepare plate for OCR ----------
def prepare_plate_for_ocr(plate_img):
    """
    Generate SINGLE optimized preprocessing for OCR (FAST).
    """
    versions = []

    # Upscale minimally (max 1.5x for speed)
    h, w = plate_img.shape[:2]
    if h > 0 and w > 0:
        target_h = max(70, h)
        scale = min(target_h / h, 1.5)  # very conservative
        if scale > 1.0:
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_LINEAR)

    # Add white border
    plate_img = cv2.copyMakeBorder(plate_img, 10, 10, 10, 10,
                                    cv2.BORDER_CONSTANT,
                                    value=[255, 255, 255])

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # SINGLE VERSION: CLAHE (most effective for plates)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    versions.append(cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR))

    return versions


# ---------- Step D: OCR with scoring ----------
# common Indian plate regex (kept for bonus scoring)
INDIAN_PLATE_PATTERN = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}")
# generic alphanumeric plate pattern for a wider range of countries
GENERIC_PLATE_PATTERN = re.compile(r"[A-Z0-9]{4,12}")

def ocr_plate(plate_versions, reader, max_candidates=2):
    """
    Run OCR on preprocessed versions (EasyOCR only for speed).
    Returns a list of up to ``max_candidates`` tuples ``(text, confidence, score)``
    sorted by score descending.
    """
    ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    all_results = []

    # EasyOCR on each version
    for i, img in enumerate(plate_versions):
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = reader.readtext(rgb, detail=1, allowlist=ALLOW,
                                          paragraph=False)
            for det in detections:
                text = re.sub(r"[^A-Z0-9]", "", det[1].upper())
                conf = det[2] if len(det) > 2 else 0.0
                if len(text) < 4:
                    continue
                # Bonus for matching known plate patterns
                bonus = 0.4 if INDIAN_PLATE_PATTERN.search(text) else 0.0
                if GENERIC_PLATE_PATTERN.search(text):
                    bonus += 0.1
                score = conf + bonus + len(text) * 0.02
                all_results.append((text, score, conf, i))
        except Exception:
            continue

    if not all_results:
        return []

    # sort and return top candidates
    all_results.sort(key=lambda x: x[1], reverse=True)
    uniq = {}
    candidates = []
    for text, score, conf, idx in all_results:
        if text not in uniq:
            uniq[text] = True
            candidates.append((text, conf, score))
            if len(candidates) >= max_candidates:
                break
    return candidates


def format_plate_text(text):
    """Try to match Indian plate pattern and format nicely."""
    m = re.search(r"([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})", text)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}"
    return text


# ---------- Full ANPR pipeline ----------
def run_plate_recognition(image_path, original, car_model, reader, fig_num):
    """Full ANPR pipeline (optimized for speed). Returns (figure, plate_text)."""

    # Step 0: Upscale if image is small (reduced target for speed)
    upscaled, scale = smart_upscale(original, target_min_dim=416)
    print(f"  Upscaled: {original.shape[:2]} -> {upscaled.shape[:2]} (x{scale:.1f})")

    # Step 1: Enhance image for plate visibility
    enhanced = enhance_for_plate(upscaled.copy())
    print("  Plate enhancement done")

    # Step 2: Detect car region (SINGLE detection for speed)
    car_crop = detect_car_region(enhanced, car_model)

    # Step 3: Find plate candidates (SINGLE search for speed)
    all_candidates = find_plate_candidates(car_crop)

    print(f"  Found {len(all_candidates)} plate candidates")

    # Sort by score and keep only top 4 to test with OCR (was 8)
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    all_candidates = all_candidates[:4]
    print(f"  Testing top {len(all_candidates)} candidates with OCR")

    # Step 4: Try OCR on top candidates only
    best_text = ""
    best_conf = 0.0
    best_plate_img = None
    extra_texts = []

    for crop, score, method in all_candidates:
        if crop is None or crop.size == 0:
            continue
        if crop.shape[0] < 5 or crop.shape[1] < 10:
            continue

        versions = prepare_plate_for_ocr(crop)
        candidates = ocr_plate(versions, reader)
        if not candidates:
            continue

        # take top candidate but remember others
        text, conf, _ = candidates[0]
        for t, c, s in candidates[1:]:
            extra_texts.append((t, c))

        if text and len(text) >= 4:
            combined = conf + (0.4 if INDIAN_PLATE_PATTERN.search(text) else 0)
            if combined > best_conf or (combined == best_conf and len(text) > len(best_text)):
                best_conf = combined
                best_text = text
                best_plate_img = crop

    plate_text = format_plate_text(best_text) if best_text else "N/A"
    # prepare string of extras
    extra_str = ""
    if extra_texts:
        uniques = []
        for t, c in extra_texts:
            if t not in uniques:
                uniques.append(t)
        extra_str = ", ".join(uniques[:2])  # show at most two extras

    # Quick fallback if still empty
    if not best_text and car_crop is not None:
        print("  Quick fallback: trying full car region")
        versions = prepare_plate_for_ocr(car_crop)
        candidates = ocr_plate(versions, reader)
        if candidates:
            best_text = candidates[0][0]
            best_conf = candidates[0][1]
            best_plate_img = car_crop
            plate_text = format_plate_text(best_text) if best_text else "N/A"

    # compute per-character ambiguity for damaged plates
    char_options = {}
    all_texts = [plate_text] + [t for t, _ in extra_texts]
    for txt in all_texts:
        for idx, ch in enumerate(txt):
            char_options.setdefault(idx, set()).add(ch)
    amb_list = []
    for idx, opts in char_options.items():
        if len(opts) > 1:
            amb_list.append(f"pos{idx+1}:{'/'.join(sorted(opts))}")
    amb_str = ", ".join(amb_list)

    print(f"  PLATE RESULT: {plate_text}  (conf: {best_conf:.2f})")
    print(f"  Best crop size: {best_plate_img.shape if best_plate_img is not None else 'None'}")
    if extra_str:
        print(f"    other possible readings: {extra_str}")
    if amb_str:
        print(f"    ambiguous characters -> {amb_str}")

    # --- Build Figure ---
    enh_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    if best_plate_img is not None and best_plate_img.size > 0:
        plate_disp = cv2.cvtColor(best_plate_img, cv2.COLOR_BGR2RGB)
        # Make plate larger and clearer
        h, w = plate_disp.shape[:2]
        if h < 150:
            scale = max(150 / h, 1.0)
            if scale > 1.0:
                plate_disp = cv2.resize(plate_disp, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_CUBIC)
    else:
        plate_disp = np.zeros((150, 400, 3), dtype=np.uint8)
        cv2.putText(plate_disp, "NO PLATE DETECTED", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), num=fig_num)
    fig.suptitle(f"PLATE RECOGNITION (ANPR)  -  {os.path.basename(image_path)}",
                 fontsize=14, fontweight="bold")

    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(enh_rgb)
    axes[1].set_title("Enhanced for Plate", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(plate_disp)
    # SIMPLIFIED: Show plate image only, minimal title
    axes[2].set_title("Detected License Plate", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    
    # Save figure to file
    fig_path = os.path.join(OUTPUT_DIR, f"plate_{os.path.basename(image_path)}")
    fig.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"  Plate figure saved to: {fig_path}")
    
    return fig, plate_text


# ===================================================================
#                    COMBINED PIPELINE
# ===================================================================
def process_image(image_path, damage_model, car_model, reader, fig_base):
    print(f"\n{'='*60}")
    print(f"  Processing: {image_path}")
    print(f"{'='*60}")

    start = time.perf_counter()
    original = cv2.imread(image_path)
    if original is None:
        print(f"  Cannot read image: {image_path}")
        return None, None

    print(f"  Image size: {original.shape}, Brightness: {assess_brightness(original):.0f}")

    print("\n  -- Pipeline 1: Damage Detection --")
    fig1 = run_damage_detection(image_path, original, damage_model, fig_num=fig_base)

    print("\n  -- Pipeline 2: Plate Recognition (ANPR) --")
    fig2, plate = run_plate_recognition(image_path, original.copy(),
                                         car_model, reader,
                                         fig_num=fig_base + 1)

    elapsed = time.perf_counter() - start
    print(f"\n  Total time: {elapsed:.2f}s")
    return fig1, fig2


# ===================================================================
#                           MAIN
# ===================================================================
def main():
    damage_model, car_model, reader = load_models()

    if len(sys.argv) > 1:
        images = sys.argv[1:]
    else:
        images = [os.path.join(SCRIPT_DIR, "i.jpg")]

    all_figs = []
    fig_counter = 1
    for img_path in images:
        if not os.path.isfile(img_path):
            print(f"File not found, skipping: {img_path}")
            continue
        f1, f2 = process_image(img_path, damage_model, car_model, reader,
                               fig_base=fig_counter)
        if f1: all_figs.append(f1)
        if f2: all_figs.append(f2)
        fig_counter += 2

    if all_figs:
        print(f"\n{'='*60}")
        print("  Figures saved to output dir. Displaying windows...")
        print(f"{'='*60}")
        # Non-blocking display with auto-close
        plt.show(block=False)
        time.sleep(20)
        plt.close('all')
        print("  Processing complete - windows closed.")
    else:
        print("No images were processed.")


if __name__ == "__main__":
    main()
