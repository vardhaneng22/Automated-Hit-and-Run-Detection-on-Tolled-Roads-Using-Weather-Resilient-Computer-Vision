import sys
import os
import cv2
import numpy as np
import re
from typing import List, Tuple
import matplotlib.pyplot as plt

# Ensure ANPR_repo is importable
repo_path = os.path.join(os.getcwd(), 'ANPR_repo')
sys.path.insert(0, repo_path)

from v8 import ANPR_V8

use_easyocr = False
reader = None
# EasyOCR initialization can be slow and heavy; enable only when environment variable ENABLE_EASYOCR=1
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


def enhance_if_lowlight(image, thresh=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray)
    if avg < thresh:
        # low light: apply CLAHE and slight gamma correction
        print(f"Low light detected (avg={avg:.1f}). Enhancing...")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # convert to BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        # mild gamma correction
        gamma = 1.1
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        enhanced_bgr = cv2.LUT(enhanced_bgr, table)
        return enhanced_bgr, True
    return image, False


def is_valid_number_plate(text: str) -> bool:
    t = text.strip().upper()
    # permissive pattern for common Indian-like plates and other variants
    pattern = r"^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4}$"
    return bool(re.match(pattern, t))


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    # keep alphanumeric only
    s = re.sub(r'[^A-Z0-9]', '', s)
    return s


def generate_variants(crop: np.ndarray) -> List[np.ndarray]:
    variants = []
    # base resized for OCR
    h, w = crop.shape[:2]
    target_w = 600
    if w < target_w:
        scale = target_w / w
        base = cv2.resize(crop, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        base = crop.copy()
    variants.append(base)

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR))

    # OTSU threshold
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR))

    # adaptive threshold
    ath = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(cv2.cvtColor(ath, cv2.COLOR_GRAY2BGR))

    # sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    variants.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

    # morphological close then open
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel2)
    variants.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))

    return variants


def ocr_candidates_from_crop(crop: np.ndarray) -> List[Tuple[str, float]]:
    """Return list of (text, score) candidates from multiple preprocess variants and OCR engines."""
    candidates = []
    variants = generate_variants(crop)
    # use global EasyOCR reader if available (avoid reinitializing heavy models)
    reader = globals().get('reader', None)

    # run EasyOCR once on a good candidate variant (CLAHE if present)
    if reader is not None:
        try:
            v_idx = 1 if len(variants) > 1 else 0
            res = reader.readtext(variants[v_idx], allowlist='.-0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
            if res:
                text_e = ''.join([r[1] for r in res])
                score_e = max([r[2] for r in res]) if res and len(res[0]) >= 3 else 0.0
                candidates.append((clean_text(text_e), float(score_e) * 100.0))
        except Exception:
            # if EasyOCR fails or is too slow, skip it and rely on pytesseract variants
            pass

    # pytesseract on all variants (faster) as fallback/multi-pass
    try:
        import pytesseract
        for v in variants:
            try:
                txt = pytesseract.image_to_string(v, config='--psm 7')
                txt = clean_text(txt)
                if txt:
                    score = len(txt)
                    candidates.append((txt, float(score)))
            except Exception:
                continue
    except Exception:
        pass

    # deduplicate keeping best score
    best = {}
    for t, s in candidates:
        if not t:
            continue
        if t not in best or best[t] < s:
            best[t] = s
    out = [(t, best[t]) for t in best]

    # boost scores for pattern matches
    out2 = []
    for t, s in out:
        if is_valid_number_plate(t):
            s = s + 1000
        out2.append((t, s))

    out2.sort(key=lambda x: x[1], reverse=True)
    return out2


def correct_ambiguities(text: str) -> str:
    if not text:
        return text
    mapping = {
        'O': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'G': '6'
    }
    # try replacing letters to digits
    t1 = ''.join([mapping.get(c, c) for c in text])
    if is_valid_number_plate(t1):
        return t1
    # try replacing digits to similar letters (reverse mapping)
    rev_map = {v: k for k, v in mapping.items()}
    t2 = ''.join([rev_map.get(c, c) for c in text])
    if is_valid_number_plate(t2):
        return t2
    return text


def enforce_strict_plate_format(text: str) -> str:
    """Enforce strict plate format LLDDLLDDDD on cleaned text by substituting ambiguous characters.
    Returns corrected plate string if successful, else empty string.
    """
    t = clean_text(text)
    template = list('LLDDLLDDDD')
    if len(t) != len(template):
        return ''

    # ambiguity mapping: prefer letter<->digit swaps when reasonable
    amb = {
        'O': ['0'], 'Q': ['0'],
        '0': ['O','Q'],
        'I': ['1','L'], 'L': ['1','I'],
        '1': ['I','L'],
        'Z': ['2'], '2': ['Z'],
        'S': ['5'], '5': ['S'],
        'B': ['8'], '8': ['B'],
        'G': ['6'], '6': ['G']
    }

    def is_letter(c):
        return 'A' <= c <= 'Z'

    def is_digit(c):
        return '0' <= c <= '9'

    # For each position build possible options
    options = []
    for idx, ch in enumerate(t):
        want = template[idx]
        opts = []
        if want == 'L':
            if is_letter(ch):
                opts.append(ch)
            # try ambiguous mappings that produce letters
            for sub in amb.get(ch, []):
                if is_letter(sub):
                    opts.append(sub)
        else:  # want digit
            if is_digit(ch):
                opts.append(ch)
            for sub in amb.get(ch, []):
                if is_digit(sub):
                    opts.append(sub)
        # also allow original if it matches after mapping from lowercase
        if not opts:
            # try mapping common misreads: treat letter that look like digit
            # if cannot produce any option, fail
            return ''
        # deduplicate
        opts = list(dict.fromkeys(opts))
        options.append(opts)

    # generate combinations with limit
    results = []
    limit = 1000

    def dfs(i, cur):
        if len(results) >= limit:
            return
        if i == len(options):
            results.append(''.join(cur))
            return
        for c in options[i]:
            cur.append(c)
            dfs(i+1, cur)
            cur.pop()

    dfs(0, [])
    # prefer original-like (minimal substitutions): choose candidate with min hamming distance
    if not results:
        return ''
    best = None
    best_dist = 1e9
    for cand in results:
        dist = sum(1 for a,b in zip(cand, t) if a != b)
        if dist < best_dist:
            best = cand
            best_dist = dist
    return best or ''

def ocr_from_crop(crop):
    if use_easyocr:
        try:
            res = reader.readtext(crop, allowlist='.-0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')
            return ''.join([r[1] for r in res])
        except Exception:
            return ''
    if use_pytesseract:
        try:
            return pytesseract.image_to_string(crop, config='--psm 8').strip()
        except Exception:
            return ''
    return ''

def process_image(path, out_name):
    img = cv2.imread(path)
    if img is None:
        print(f"Image not found: {path}")
        return
    # determine low-light and optionally enhance
    proc_img, enhanced = enhance_if_lowlight(img)
    plates, out_img = model.detect(proc_img, threshold=0.3)
    print(f"{path}: detected {len(plates)} plates")
    # sort by confidence (if available) and limit number of plates processed
    def conf_of(p):
        try:
            return float(p[4])
        except Exception:
            return 0.0
    plates_sorted = sorted(plates, key=conf_of, reverse=True)
    max_process = 5
    for i, plate in enumerate(plates_sorted[:max_process]):
        try:
            x1,y1,x2,y2,conf = plate[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except Exception:
            print('Unexpected plate format:', plate)
            continue
        # ensure valid crop coordinates
        x1c, y1c = max(x1,0), max(y1,0)
        x2c, y2c = max(x2, x1c+1), max(y2, y1c+1)
        crop = out_img[y1c:y2c, x1c:x2c]
        # try our multi-variant OCR to get best candidate
        h, w = crop.shape[:2]
        if h < 8 or w < 24:
            print(f" Skipping tiny crop {w}x{h}")
            best_text = ''
        else:
            candidates = ocr_candidates_from_crop(crop)
            best_text = candidates[0][0] if candidates else ''
            # enforce strict format: try corrected ambiguity then strict positional enforcement
            corrected = correct_ambiguities(best_text)
            strict = enforce_strict_plate_format(corrected)
            if strict:
                best_text = strict
        print(f" Plate {i+1}: conf={conf} text='{best_text}'")
        # draw bbox and label with filled background for readability
        h_img, w_img = out_img.shape[:2]
        base = min(h_img, w_img)
        thickness = max(2, int(base / 500))
        font_scale = max(1.0, base / 1500)
        label = best_text if best_text else f"conf:{conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(2, thickness))
        # background rectangle
        lbx, lby = x1, max(y1 - lh - 10, 0)
        cv2.rectangle(out_img, (lbx, lby), (lbx + lw + 6, lby + lh + 6), (0,255,0), -1)
        cv2.putText(out_img, label, (lbx + 3, lby + lh + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), max(1, thickness//2))
        cv2.rectangle(out_img, (x1,y1), (x2,y2), (0,255,0), thickness)
    # Show popup window using matplotlib (works where OpenCV GUI may not be available)
    try:
        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        proc_rgb = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)
        res_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(proc_rgb)
        axes[1].set_title('Processed (enhanced if low-light)')
        axes[1].axis('off')

        axes[2].imshow(res_rgb)
        axes[2].set_title('Detection + OCR')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not open matplotlib popup ({e}). Run this script where a display is available.")

if __name__ == '__main__':
    images = [
        os.path.join(os.getcwd(), 'i.jpg'),
        os.path.join(os.getcwd(), 'i2.jpg'),
        os.path.join(os.getcwd(), 'i3.jpg'),
        os.path.join(os.getcwd(), 'car3.png'),
    ]
    os.makedirs('outputs', exist_ok=True)
    for idx, p in enumerate(images, start=1):
        out_path = os.path.join('outputs', f'repo_test_out_{idx}.jpg')
        process_image(p, out_path)
