import os
import sys
import time
import re
import requests
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
import easyocr

# ---------- Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAINED_WEIGHTS_URL = "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector/raw/main/trained.pt"
TRAINED_WEIGHTS_FILE = os.path.join(SCRIPT_DIR, 'trained.pt')
CAR_MODEL_FILE = os.path.join(SCRIPT_DIR, 'yolov8n.pt')

# ---------- Utilities (from final.py)
class ContrastEnhancer:
    def remove_fog_veil(self, image):
        img_float = image.astype(np.float32)
        min_val = np.percentile(img_float, 2)
        max_val = np.percentile(img_float, 99)
        if max_val - min_val > 0:
            img_stretched = (img_float - min_val) * (255.0 / (max_val - min_val))
        else:
            img_stretched = img_float
        img_stretched = np.clip(img_stretched, 0, 255).astype(np.uint8)
        return img_stretched

    def sharpen_edges(self, image):
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        unsharp_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0, image)
        return unsharp_image

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        de_fogged = self.remove_fog_veil(img)
        sharpened = self.sharpen_edges(de_fogged)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final, img

# ---------- Plate helpers (from new.py)
def enhance_image_retinex(image):
    img = image.astype(np.float32) + 1.0
    b, g, r = cv2.split(img)
    def retinex(channel):
        blur = cv2.GaussianBlur(channel, (0,0), 30)
        return np.log(channel) - np.log(blur + 1)
    r = retinex(r); g = retinex(g); b = retinex(b)
    merged = cv2.merge((b,g,r))
    merged = cv2.normalize(merged, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab = cv2.cvtColor(merged, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(3.0,(8,8))
    l = clahe.apply(l)
    final = cv2.merge((l,a,b))
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)
    return final

def detect_car_in_image(yolo_car, image):
    res = yolo_car(image)[0]
    for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
        if int(cls) == 2:  # car
            x1,y1,x2,y2 = map(int,box)
            return image[y1:y2, x1:x2], (x1,y1)
    return None, None

def detect_plate_from_car_simple(car_img):
    gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        ratio = w/float(h) if h>0 else 0
        if 2 <= ratio <= 6 and w>80 and h>20:
            px,py = int(w*0.15), int(h*0.25)
            x1 = max(0,x-px); y1 = max(0,y-py)
            x2 = min(car_img.shape[1], x+w+px); y2 = min(car_img.shape[0], y+h+py)
            return car_img[y1:y2, x1:x2], (x1,y1,x2-x1,y2-y1)
    return None, None

def recognize_plate_text(reader, plate_img):
    if plate_img is None or plate_img.size==0:
        return ""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    results = reader.readtext(gray, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', detail=0)
    text = "".join(results)
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return text

# ---------- Main combined flow
def ensure_weights():
    if not os.path.exists(TRAINED_WEIGHTS_FILE):
        print('Downloading damage weights...')
        r = requests.get(TRAINED_WEIGHTS_URL, stream=True); r.raise_for_status()
        with open(TRAINED_WEIGHTS_FILE, 'wb') as f:
            for chunk in r.iter_content(8192): f.write(chunk)

def run_combined(image_path):
    ensure_weights()
    damage_model = YOLO(TRAINED_WEIGHTS_FILE)
    car_model = YOLO(CAR_MODEL_FILE)
    reader = easyocr.Reader(['en'], gpu=False)
    enhancer = ContrastEnhancer()
    enhanced_damage, original = enhancer.process(image_path)
    if enhanced_damage is None:
        print('Cannot read image'); return
    results = damage_model.predict(enhanced_damage, conf=0.35)
    res_plotted = None
    for r in results:
        res_plotted = r.plot(); res_obj = r; break
    image = cv2.imread(image_path)
    plate_enh = enhance_image_retinex(image)
    car_crop, offset = detect_car_in_image(car_model, plate_enh)
    plate_crop = None; box = None
    if car_crop is not None:
        plate_crop, local_box = detect_plate_from_car_simple(car_crop)
        if plate_crop is not None:
            ox,oy = offset; x,y,w,h = local_box; box = (x+ox,y+oy,w,h)
    if plate_crop is None:
        h,w = plate_enh.shape[:2]
        y1,y2 = int(h*0.55), int(h*0.80); x1,x2 = int(w*0.15), int(w*0.85)
        plate_crop = plate_enh[y1:y2, x1:x2]; box=(x1,y1,x2-x1,y2-y1)
    plate_text = recognize_plate_text(reader, plate_crop)
    fig1, axes1 = plt.subplots(1,3, figsize=(18,6))
    axes1[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); axes1[0].set_title('Original'); axes1[0].axis('off')
    axes1[1].imshow(cv2.cvtColor(enhanced_damage, cv2.COLOR_BGR2RGB)); axes1[1].set_title('Enhanced Damage'); axes1[1].axis('off')
    if res_plotted is not None:
        axes1[2].imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)); axes1[2].set_title('YOLO Damage'); axes1[2].axis('off')
    else:
        axes1[2].imshow(np.zeros((100,200,3),dtype=np.uint8)); axes1[2].set_title('No Detections'); axes1[2].axis('off')
    fig1.tight_layout(); fig1_path = os.path.join(OUTPUT_DIR, 'damage_'+os.path.basename(image_path)); fig1.savefig(fig1_path)
    fig2, axes2 = plt.subplots(1,3, figsize=(18,6))
    axes2[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); axes2[0].set_title('Original'); axes2[0].axis('off')
    axes2[1].imshow(cv2.cvtColor(plate_enh, cv2.COLOR_BGR2RGB)); axes2[1].set_title('Enhanced for Plate'); axes2[1].axis('off')
    if plate_crop is not None:
        axes2[2].imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)); axes2[2].set_title(f'Plate Crop: {plate_text}'); axes2[2].axis('off')
    else:
        axes2[2].imshow(np.zeros((120,320,3),dtype=np.uint8)); axes2[2].set_title('No Plate'); axes2[2].axis('off')
    fig2.tight_layout(); fig2_path = os.path.join(OUTPUT_DIR, 'plate_'+os.path.basename(image_path)); fig2.savefig(fig2_path)
    if plate_crop is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'plate_crop_'+os.path.basename(image_path)), plate_crop)
    plt.show(block=False)
    print('Figures saved to', OUTPUT_DIR)
    time.sleep(15)
    plt.close('all')

if __name__ == '__main__':
    imgs = []
    if len(sys.argv)>1:
        imgs = sys.argv[1:]
    else:
        imgs = [os.path.join(SCRIPT_DIR, 'i.jpg'), os.path.join(SCRIPT_DIR, 'i2.jpg')]
    for im in imgs:
        if os.path.exists(im):
            print('Processing', im)
            run_combined(im)
        else:
            print('Image not found:', im)
