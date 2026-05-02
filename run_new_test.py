import sys
import os
import cv2
import numpy as np
import re
from typing import List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure ANPR_repo is importable
repo_path = os.path.join(os.getcwd(), 'ANPR_repo')
sys.path.insert(0, repo_path)

from v8 import ANPR_V8

use_easyocr = False
reader = None

# EasyOCR initialization can be slow and heavy
if os.environ.get('ENABLE_EASYOCR') == '1':
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        use_easyocr = True
    except Exception:
        use_easyocr = False

try:
    import pytesseract
    use_pytesseract = True
except Exception:
    use_pytesseract = False


model = ANPR_V8(os.path.join(repo_path, 'models', 'anpr_v8.pt'))


# --------------------------------------------------
# LOW LIGHT ENHANCEMENT
# --------------------------------------------------

def enhance_if_lowlight(image, thresh=100):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray)

    if avg < thresh:

        print(f"Low light detected (avg={avg:.1f}). Enhancing...")

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        gamma = 1.1
        invGamma = 1.0 / gamma

        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype('uint8')

        enhanced_bgr = cv2.LUT(enhanced_bgr, table)

        return enhanced_bgr, True

    return image, False


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def is_valid_number_plate(text: str) -> bool:

    t = text.strip().upper()

    pattern = r"^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}$"

    return bool(re.match(pattern, t))


def clean_text(s: str) -> str:

    if not s:
        return ""

    s = s.upper()

    s = re.sub(r'[^A-Z0-9]', '', s)

    return s


# --------------------------------------------------
# OCR VARIANT GENERATION
# --------------------------------------------------

def generate_variants(crop: np.ndarray) -> List[np.ndarray]:

    variants = []

    h, w = crop.shape[:2]

    target_w = 600

    if w < target_w:

        scale = target_w / w

        base = cv2.resize(crop, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    else:

        base = crop.copy()

    variants.append(base)

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    variants.append(cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR))

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))

    ath = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,11,2
    )

    variants.append(cv2.cvtColor(ath, cv2.COLOR_GRAY2BGR))

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

    sharp = cv2.filter2D(gray,-1,kernel)

    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2)

    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel2)

    variants.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))

    return variants


# --------------------------------------------------
# OCR ENGINE
# --------------------------------------------------

def ocr_candidates_from_crop(crop: np.ndarray) -> List[Tuple[str, float]]:

    candidates = []

    variants = generate_variants(crop)

    reader = globals().get('reader', None)

    if reader is not None:

        try:

            v_idx = 1 if len(variants) > 1 else 0

            res = reader.readtext(
                variants[v_idx],
                allowlist='.-0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
            )

            if res:

                text_e = ''.join([r[1] for r in res])

                score_e = max([r[2] for r in res])

                candidates.append((clean_text(text_e), float(score_e)*100))

        except Exception:
            pass

    try:

        for v in variants:

            txt = pytesseract.image_to_string(v, config='--psm 7')

            txt = clean_text(txt)

            if txt:

                candidates.append((txt, float(len(txt))))

    except Exception:
        pass


    best = {}

    for t,s in candidates:

        if t not in best or best[t] < s:
            best[t] = s

    out = [(t,best[t]) for t in best]

    out.sort(key=lambda x:x[1], reverse=True)

    return out


# --------------------------------------------------
# MAIN PROCESS
# --------------------------------------------------

def process_image(path, out_name, show_cropped_plate=True):

    os.makedirs("outputs", exist_ok=True)

    img = cv2.imread(path)

    if img is None:
        print("Image not found:", path)
        return None

    proc_img, enhanced = enhance_if_lowlight(img)

    plates, out_img = model.detect(proc_img, threshold=0.3)

    print(f"{path}: detected {len(plates)} plates")

    top_crop = None
    top_best_text = ""

    for i, plate in enumerate(plates):

        x1,y1,x2,y2,conf = plate[:5]

        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

        crop = out_img[y1:y2,x1:x2]

        if crop.size == 0:
            continue

        candidates = ocr_candidates_from_crop(crop)

        best_text = candidates[0][0] if candidates else ""

        print(f"Plate {i+1}: conf={conf} text='{best_text}'")

        cv2.rectangle(out_img,(x1,y1),(x2,y2),(0,255,0),2)

        if i == 0:

            top_crop = crop.copy()

            top_best_text = best_text


    # --------------------------------------------------
    # SAVE ANPR DETECTION IMAGE
    # --------------------------------------------------

    anpr_output_path = out_name if out_name else os.path.join("outputs", "anpr_result.jpg")

    cv2.imwrite(anpr_output_path, out_img)


    # --------------------------------------------------
    # SAVE CROPPED PLATE
    # --------------------------------------------------

    out_dir = os.path.dirname(anpr_output_path) or "outputs"
    base = os.path.splitext(os.path.basename(anpr_output_path))[0]
    crop_output_path = os.path.join(out_dir, f"{base}_plate_crop.jpg")

    if top_crop is not None:

        cv2.imwrite(crop_output_path, top_crop)


    # --------------------------------------------------
    # DASHBOARD VISUALIZATION IMAGE (REPLACES POPUP)
    # --------------------------------------------------

    dashboard_preview_path = os.path.join(out_dir, f"{base}_anpr_preview.jpg")
    try:

        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        proc_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)

        if top_crop is not None:

            crop_rgb = cv2.cvtColor(top_crop, cv2.COLOR_BGR2RGB)

        else:

            crop_rgb = np.full((300,300,3),200,dtype=np.uint8)


        fig, axes = plt.subplots(1,3,figsize=(18,6))

        axes[0].imshow(orig_rgb)
        axes[0].set_title("Original")

        axes[1].imshow(proc_rgb)
        axes[1].set_title("Processed")

        axes[2].imshow(crop_rgb)
        axes[2].set_title("Cropped Plate")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(dashboard_preview_path)

        plt.close()

    except Exception as e:

        print("Visualization failed:", e)


    # --------------------------------------------------
    # RETURN DATA FOR DASHBOARD
    # --------------------------------------------------

    return {

        "plate_number": top_best_text,

        "plate_crop": crop_output_path,

        "anpr_image": anpr_output_path,

        "dashboard_preview": dashboard_preview_path

    }



# --------------------------------------------------
# TEST MODE
# --------------------------------------------------

if __name__ == "__main__":

    images = [

        os.path.join(os.getcwd(),'i.jpg'),
        os.path.join(os.getcwd(),'i2.jpg'),
        os.path.join(os.getcwd(),'i3.jpg'),
        os.path.join(os.getcwd(),'car3.png')

    ]

    os.makedirs("outputs", exist_ok=True)

    for idx,p in enumerate(images,start=1):

        out_path = os.path.join("outputs", f"repo_test_out_{idx}.jpg")

        process_image(p, out_path)