import cv2
import final
import run_new_test
from generate_report import generate_claim_report
import os
import shutil

from forensic_analyzer import *
from groq_ai import ai_forensic_analysis
import concurrent.futures


def run_full_analysis(image_path, output_prefix=None, generate_pdf=True, generate_ai=True):

    os.makedirs("outputs", exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_damage = executor.submit(final.process_system, image_path)
        future_anpr = executor.submit(
            run_new_test.process_image,
            image_path,
            os.path.join("outputs", f"{output_prefix}_anpr.jpg") if output_prefix else "outputs/anpr_result.jpg"
        )
        damage = future_damage.result()
        anpr = future_anpr.result()

    img = cv2.imread(image_path)

    damages = damage.get("damages", [])

    if len(damages) > 0:
        bbox = damages[0]["bbox"]
    else:
        bbox = [0,0,0,0]

    source = estimate_impact_source(bbox, img.shape[0])

    direction = detect_scratch_direction(img)

    severity = severity_score(damages, img.shape)

    plate = anpr.get("plate_number","Not detected")

    plate_crop_img = None
    plate_crop_path = anpr.get("plate_crop")
    if plate_crop_path and cv2.imread(plate_crop_path) is not None:
        plate_crop_img = cv2.imread(plate_crop_path)

    def _run_ai():
        if generate_ai:
            if severity == 0:
                return "Vehicle is cleared. No damage detected. AI analysis not required."
            return ai_forensic_analysis({
                "plate": plate,
                "damage": "vehicle damage detected" if severity > 0 else "no damage detected",
                "source": source,
                "direction": direction,
                "severity": severity
            })
        return ""

    def _run_pdf():
        if generate_pdf:
            report_pdf_path = generate_claim_report(
                image_path=image_path,
                annotated_image=cv2.imread(damage.get("annotated_image", "outputs/damage_result.jpg")),
                damages=damages,
                plate_number=plate,
                plate_crop=plate_crop_img,
                output_dir="outputs",
            )
            return os.path.basename(report_pdf_path)
        return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_ai = executor.submit(_run_ai)
        f_pdf = executor.submit(_run_pdf)
        ai_report = f_ai.result()
        report_pdf_name = f_pdf.result()

    if output_prefix:
        damage_src = damage.get("annotated_image", "outputs/damage_result.jpg")
        damage_dst = os.path.join("outputs", f"{output_prefix}_damage.jpg")
        try:
            shutil.copyfile(damage_src, damage_dst)
            damage["annotated_image"] = damage_dst
        except Exception:
            damage["annotated_image"] = damage_src

    if output_prefix:
        anpr_src = anpr.get("anpr_image", "outputs/anpr_result.jpg")
        anpr_dst = os.path.join("outputs", f"{output_prefix}_anpr_result.jpg")
        try:
            shutil.copyfile(anpr_src, anpr_dst)
            anpr["anpr_image"] = anpr_dst
        except Exception:
            anpr["anpr_image"] = anpr_src

        plate_crop_src = anpr.get("plate_crop", "outputs/plate_crop.jpg")
        plate_crop_dst = os.path.join("outputs", f"{output_prefix}_plate_crop.jpg")
        try:
            shutil.copyfile(plate_crop_src, plate_crop_dst)
            anpr["plate_crop"] = plate_crop_dst
        except Exception:
            anpr["plate_crop"] = plate_crop_src

    damage_image = damage.get("annotated_image", "outputs/damage_result.jpg")
    anpr_image = anpr.get("anpr_image", "outputs/anpr_result.jpg")
    plate_crop = anpr.get("plate_crop", "outputs/plate_crop.jpg")

    damage_image = damage_image.replace('\\', '/')
    anpr_image = anpr_image.replace('\\', '/')
    plate_crop = plate_crop.replace('\\', '/')

    return {
    "plate": plate,
    "source": source,
    "direction": direction,
    "severity": severity,
    "ai_report": ai_report,
    "damage_image": damage_image,
    "anpr_image": anpr_image,
    "plate_crop": plate_crop,
    "report_pdf_name": report_pdf_name
}