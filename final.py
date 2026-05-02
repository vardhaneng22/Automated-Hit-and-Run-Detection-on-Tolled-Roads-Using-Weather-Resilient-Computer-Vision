import os
import sys
import requests
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

REPO_URL = "https://github.com/ReverendBayes/YOLO11m-Car-Damage-Detector/raw/main/trained.pt"
MODEL_FILENAME = "trained.pt"

OUTPUT_IMAGE = "outputs/damage_result.jpg"


# ==========================================
# IMAGE ENHANCEMENT
# ==========================================

class ContrastEnhancer:

    def __init__(self):
        print("📉 De-Fogger Initialized (Crushing Blacks)")


    def remove_fog_veil(self, image):

        img_float = image.astype(np.float32)

        min_val = np.percentile(img_float, 2)
        max_val = np.percentile(img_float, 99)
        
        dynamic_range = max_val - min_val

        if dynamic_range > 0:
            multiplier = 255.0 / dynamic_range
            # Prevent extreme amplification on very dark images to avoid noise (false positives)
            multiplier = min(multiplier, 3.0) 
            img_stretched = (img_float - min_val) * multiplier
        else:
            img_stretched = img_float

        img_stretched = np.clip(img_stretched, 0, 255).astype(np.uint8)

        return img_stretched


    def sharpen_edges(self, image):

        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        unsharp_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

        return unsharp_image


    def process(self, image_path):

        img = cv2.imread(image_path)

        if img is None:
            return None, None

        # Step 1 – De-fog
        de_fogged = self.remove_fog_veil(img)

        # Step 2 – Sharpen
        sharpened = self.sharpen_edges(de_fogged)

        # Step 3 – CLAHE contrast boost
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))

        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return final, img


# ==========================================
# MODEL DOWNLOAD
# ==========================================

def download_weights(url, filename):

    if os.path.exists(filename):
        return

    print("⬇️ Downloading pre-trained weights...")

    try:

        response = requests.get(url, stream=True)

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    except Exception as e:

        print("❌ Error downloading weights:", e)
        sys.exit(1)


# ==========================================
# GLOBALS FOR FAST INFERENCE
# ==========================================
os.makedirs("outputs", exist_ok=True)
download_weights(REPO_URL, MODEL_FILENAME)
print("🚀 Loading YOLO11m Damage Detector (GLOBAL)...")
_cached_enhancer = ContrastEnhancer()
_cached_model = YOLO(MODEL_FILENAME)

# ==========================================
# DAMAGE DETECTION PIPELINE
# ==========================================

def process_system(image_path):

    os.makedirs("outputs", exist_ok=True)

    enhancer = _cached_enhancer
    model = _cached_model

    print(f"🌊 Step 1: De-Fogging {image_path}...")

    clean_image, original_image = enhancer.process(image_path)

    if clean_image is None:
        print("❌ Error reading image")
        return None


    print("🔍 Step 2: Detecting Damage...")

    results = model.predict(clean_image, save=False, conf=0.35)


    damage_data = {
        "annotated_image": OUTPUT_IMAGE,
        "damages": []
    }


    for result in results:
        
        if 99 not in result.names:
            result.names[99] = "severe_destruction"
        if 99 not in model.names:
            model.names[99] = "severe_destruction"
            
        img_h, img_w = clean_image.shape[:2]
        img_area = img_h * img_w
        
        total_damage_area = sum([(box.xyxy[0][2] - box.xyxy[0][0]).item() * (box.xyxy[0][3] - box.xyxy[0][1]).item() for box in result.boxes])
        total_ratio = total_damage_area / float(img_area + 1e-6)
        
        new_data = result.boxes.data.clone()
        has_changed = False
        
        structural_count = sum(1 for box in result.boxes if result.names.get(int(box.cls[0].item()), "") in ["dent", "shattered_glass", "severe_destruction"])
        
        # A scene is wrecked if there are many structural breakages (e.g. 2 dents + shattered glass) OR huge overall damage area
        scene_is_wrecked = (structural_count >= 3) or (total_ratio > 0.15)
        
        for i, box in enumerate(result.boxes):
            w = (box.xyxy[0][2] - box.xyxy[0][0]).item()
            h = (box.xyxy[0][3] - box.xyxy[0][1]).item()
            box_area = w * h
            box_ratio = box_area / float(img_area + 1e-6)
            
            cls_id = int(box.cls[0].item())
            cls_name = result.names.get(cls_id, "")
            
            # Shattered glass ALWAYS remains shattered glass.
            # Ripped off surfaces / crushed structural parts become severe_destruction.
            if cls_name in ["dent", "scratch"]:
                # If the car is generally wrecked, substantial structural bends (>2% area) are designated as severe destruction
                if scene_is_wrecked and box_ratio > 0.02:
                    new_data[i, 5] = 99.0
                    has_changed = True
                # If the car is otherwise fine, ONLY an absolutely massive gaping hole (>10% area) is severe destruction
                elif box_ratio > 0.10:
                    new_data[i, 5] = 99.0
                    has_changed = True
                    
        if has_changed:
            from ultralytics.engine.results import Boxes
            result.boxes = Boxes(new_data, result.orig_shape)

        annotated = result.plot()

        # Convert to RGB for plotting
        orig_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)
        res_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(orig_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(clean_rgb)
        axes[1].set_title("De-Fogged + Enhanced")
        axes[1].axis('off')

        axes[2].imshow(res_rgb)
        axes[2].set_title("Damage Detection")
        axes[2].axis('off')

        plt.tight_layout()

        plt.savefig(OUTPUT_IMAGE)

        plt.close()


        print("\n--- Damage Report ---")

        if len(result.boxes) == 0:

            print("No damage detected.")

        else:

            for box in result.boxes:

                cls = result.names.get(int(box.cls[0]), "unknown")
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                print(f"• {cls.upper()} (Conf: {conf:.2f})")

                damage_data["damages"].append({
                    "type": cls,
                    "confidence": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })


    # If for some reason nothing was saved
    if not os.path.exists(OUTPUT_IMAGE):

        cv2.imwrite(OUTPUT_IMAGE, clean_image)


    return damage_data


# ==========================================
# CLI ENTRY
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        process_system(sys.argv[1])

    else:

        path = input("Enter image path: ").strip('"')

        if os.path.exists(path):

            process_system(path)

        else:

            print("Image not found.")