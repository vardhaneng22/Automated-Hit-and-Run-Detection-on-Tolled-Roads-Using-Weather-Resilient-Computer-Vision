"""Run both Damage Detection (from final.py) and ANPR (from run_new_test.py) on a single image.

Usage:
    python run_both.py /path/to/image.jpg

This will display two popups (matplotlib): first damage detection, then ANPR.
After both complete, a PDF insurance claim report is auto-generated in outputs/.
If a model required by `final.py` is missing it will attempt to download it (as in final.py).
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        img_path = input("Enter image path: ").strip('"')
    else:
        img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        return

    damage_result = None
    anpr_result = None

    # Run damage detection from final.py
    try:
        import final
        print("Running damage detection (final.py)...")
        damage_result = final.process_system(img_path)
    except Exception as e:
        print("Damage detection failed:", e)

    # Run ANPR from run_new_test.py
    try:
        import run_new_test
        print("Running ANPR (run_new_test.py)...")
        out_name = os.path.join(os.getcwd(), 'outputs', 'run_both_anpr_out.jpg')
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        # show_cropped_plate=True to ensure the ANPR popup is 3-panel with cropped plate
        anpr_result = run_new_test.process_image(img_path, out_name, show_cropped_plate=True)
    except Exception as e:
        print("ANPR failed:", e)

    # Generate PDF Insurance Claim Report
    try:
        from generate_report import generate_claim_report
        print("\n📄 Generating Insurance Claim Report...")
        generate_claim_report(
            image_path=img_path,
            annotated_image=damage_result.get("annotated_image") if damage_result else None,
            damages=damage_result.get("damages", []) if damage_result else [],
            plate_number=anpr_result.get("plate_number", "") if anpr_result else "",
            plate_crop=anpr_result.get("plate_crop") if anpr_result else None,
            output_dir="outputs",
        )
    except Exception as e:
        print("Report generation failed:", e)

if __name__ == '__main__':
    main()

