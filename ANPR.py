import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import os
import sys

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure you've downloaded the correct model

# Helper function to enhance image contrast in low light conditions
def enhance_contrast(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the histogram of the image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # If the average pixel value is low (indicating low light), apply contrast enhancement
    avg_pixel_value = np.mean(gray)
    if avg_pixel_value < 100:  # Threshold for low light
        print("Low light detected. Enhancing contrast...")
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)
        return cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    return image  # Return original image if no enhancement needed

# Function to check if the detected text matches the Indian Number Plate format
def is_valid_number_plate(text):
    # Pattern for valid Indian Number Plates: [State Code] [RTO District Code] [Alphabet Series] [Unique Number]
    # Example: MH 12 AB 1234
    # Format: 2-letter state code, 2-digit district code, 2-letter alphabet series, 4-digit number
    import re
    pattern = r"^[A-Z]{2}\s\d{2}\s[A-Z]{2}\s\d{4}$"
    return bool(re.match(pattern, text.strip()))

import easyocr
import re

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Load input image
image_path = sys.argv[1] if len(sys.argv) > 1 else "i2.jpg"
filename = os.path.basename(image_path).split('.')[0]
input_image = cv2.imread(image_path) 
if input_image is None:
    print(f"Input image '{image_path}' not found.")
    sys.exit(1)

# Perform YOLOv8 inference to detect vehicles
results = model(input_image, conf=0.3)

def get_ocr_text(img):
    """Try Tesseract first, then EasyOCR."""
    text = ""
    # Tesseract Attempt
    try:
        # Check if tesseract_cmd is set or in PATH
        if not pytesseract.pytesseract.tesseract_cmd:
            common_paths = [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                            r"C:\Users\GCP Intern 2\AppData\Local\Tesseract-OCR\tesseract.exe"]
            for p in common_paths:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break
        
        text = pytesseract.image_to_string(img, config='--psm 7').strip()
        if text: print(f"Tesseract raw: {text}")
    except Exception as e:
        print(f"Tesseract failed: {e}")

    # EasyOCR Fallback/Refinement
    try:
        results = reader.readtext(img, detail=0)
        easy_text = "".join(results)
        print(f"EasyOCR raw: {easy_text}")
        if not text or len(easy_text) > len(text):
            text = easy_text
    except Exception as e:
        print(f"EasyOCR failed: {e}")
    
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# Constants for detection
vehicle_classes = {2, 3, 5, 7} # car, motorcycle, bus, truck
plate_found = False
os.makedirs("outputs/debug", exist_ok=True)

for result in results:
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        if int(cls) in vehicle_classes:
            # ... existing vehicle logic ...
            pass
    if plate_found: break

# ROI Fallback if nothing found
if not plate_found:
    h_img, w_img = input_image.shape[:2]
    y1, y2 = int(h_img * 0.55), int(h_img * 0.80)
    x1, x2 = int(w_img * 0.20), int(w_img * 0.80)
    plate_crop = input_image[y1:y2, x1:x2]
    crop_path = f"outputs/debug/{filename}_ANPR_plate_fallback.jpg"
    cv2.imwrite(crop_path, plate_crop)
    print(f"Applied ROI Fallback and saved to {crop_path}")
    plate_text = get_ocr_text(plate_crop)
    if plate_text:
        print(f"Detected (Fallback): {plate_text}")
        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(input_image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save result
res_path = f"outputs/debug/{filename}_ANPR_result.jpg"
cv2.imwrite(res_path, input_image)
print(f"Saved visualization to {res_path}")
