"""
Auto-Generate a PDF Insurance Claim Report.

After running damage detection and ANPR, call `generate_claim_report()`
to produce a polished, printable PDF with all findings.
"""

import os
import uuid
import tempfile
from datetime import datetime

import cv2
from fpdf import FPDF


class ClaimReportPDF(FPDF):
    """Custom FPDF subclass with branded header/footer."""

    def __init__(self, claim_id, timestamp_str):
        super().__init__()
        self.claim_id = claim_id
        self.timestamp_str = timestamp_str

    # ── Header ──────────────────────────────────────────────
    def header(self):
        # Brand bar
        self.set_fill_color(15, 30, 65)  # dark navy
        self.rect(0, 0, 210, 28, 'F')

        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(255, 255, 255)
        self.set_y(5)
        self.cell(0, 10, 'VEHICLE INSURANCE CLAIM REPORT', 0, 1, 'C')

        self.set_font('Helvetica', '', 9)
        self.set_text_color(180, 200, 230)
        info_line = 'Claim ID: ' + self.claim_id + '     |     Generated: ' + self.timestamp_str
        self.cell(0, 6, info_line, 0, 1, 'C')
        self.ln(8)

    # ── Footer ──────────────────────────────────────────────
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}   |   Auto-generated - Confidential',
                  align='C')

    # ── helpers ─────────────────────────────────────────────
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(15, 30, 65)
        self.cell(0, 10, title, 0, 1)
        # underline bar
        self.set_draw_color(30, 80, 160)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), 200, self.get_y())
        self.ln(4)

    def key_value(self, key, value):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(60, 60, 60)
        self.cell(45, 7, key + ':', 0, 0)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, str(value), 0, 1)


def _save_temp_image(img_bgr, prefix='rpt_'):
    """Save a BGR numpy array as a temp JPEG and return the path."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix='.jpg')
    os.close(fd)
    cv2.imwrite(path, img_bgr)
    return path


def _severity_label(damages):
    """Simple heuristic severity from count & types."""
    if not damages:
        return 'NONE', (34, 139, 34)  # green
    n = len(damages)
    avg_conf = sum(d.get('confidence', 0) for d in damages) / n
    if n >= 3 or avg_conf >= 0.80:
        return 'HIGH', (200, 30, 30)
    if n >= 2 or avg_conf >= 0.60:
        return 'MODERATE', (210, 140, 0)
    return 'LOW', (34, 139, 34)


def _cost_estimate(damages):
    """Rough INR cost lookup per damage type."""
    cost_map = {
        'dent':       (2000, 5000),
        'scratch':    (1000, 3000),
        'crack':      (3000, 8000),
        'shatter':    (5000, 15000),
        'broken':     (4000, 12000),
        'flat tire':  (1500, 4000),
    }
    lo_total, hi_total = 0, 0
    for d in damages:
        t = d.get('type', '').lower()
        lo, hi = cost_map.get(t, (1000, 5000))
        lo_total += lo
        hi_total += hi
    return lo_total, hi_total


# ════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════
def generate_claim_report(
    image_path,
    annotated_image=None,
    damages=None,
    plate_number='',
    plate_crop=None,
    output_dir='outputs',
):
    """
    Generate a professional PDF claim report.

    Parameters
    ----------
    image_path       : str          – path to the original car image
    annotated_image  : np.ndarray   – BGR image with damage bounding boxes drawn
    damages          : list[dict]   – [{"type": str, "confidence": float, "bbox": tuple}, ...]
    plate_number     : str          – recognised plate text (may be empty)
    plate_crop       : np.ndarray   – BGR cropped plate image or None
    output_dir       : str          – folder for the PDF

    Returns
    -------
    str – absolute path to the saved PDF
    """
    if damages is None:
        damages = []

    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    timestamp_str = now.strftime('%d %b %Y, %I:%M %p')
    short_id = uuid.uuid4().hex[:6].upper()
    claim_id = 'CLM-' + now.strftime('%Y%m%d') + '-' + short_id

    pdf = ClaimReportPDF(claim_id, timestamp_str)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    temp_files = []

    # ── 1  VEHICLE INFORMATION ──────────────────────────────
    pdf.section_title('1.  Vehicle Information')
    pdf.key_value('License Plate', plate_number if plate_number else 'Not detected')
    pdf.key_value('Image Source', os.path.basename(image_path))
    pdf.key_value('Report Date', timestamp_str)
    pdf.ln(3)

    # plate crop image (small)
    if plate_crop is not None:
        tmp = _save_temp_image(plate_crop, 'plate_')
        temp_files.append(tmp)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, 'Detected plate region:', 0, 1)
        try:
            h_crop, w_crop = plate_crop.shape[:2]
            display_w = min(60, w_crop * 0.3)
            pdf.image(tmp, x=pdf.l_margin, w=display_w)
        except Exception:
            pass
        pdf.ln(4)

    # ── 2  DAMAGE ANALYSIS ──────────────────────────────────
    pdf.section_title('2.  Damage Analysis')

    severity, sev_color = _severity_label(damages)
    pdf.key_value('Damages Found', str(len(damages)))
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(*sev_color)
    pdf.cell(45, 7, 'Severity:', 0, 0)
    pdf.cell(0, 7, severity, 0, 1)
    pdf.set_text_color(30, 30, 30)

    if damages:
        lo, hi = _cost_estimate(damages)
        pdf.set_font('Helvetica', '', 10)
        pdf.key_value('Est. Repair Cost', f'Rs. {lo:,} - Rs. {hi:,}')
    pdf.ln(3)

    # damage table
    if damages:
        pdf.set_font('Helvetica', 'B', 10)
        # table header
        pdf.set_fill_color(30, 60, 120)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(10, 8, '#', border=1, align='C', fill=True)
        pdf.cell(45, 8, 'Damage Type', border=1, align='C', fill=True)
        pdf.cell(35, 8, 'Confidence', border=1, align='C', fill=True)
        pdf.cell(80, 8, 'Bounding Box (x1, y1, x2, y2)', border=1, align='C', fill=True, ln=1)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 30, 30)
        for idx, d in enumerate(damages, 1):
            fill = idx % 2 == 0
            if fill:
                pdf.set_fill_color(235, 240, 250)
            dtype = d.get('type', 'Unknown').upper()
            conf = f"{d.get('confidence', 0) * 100:.1f}%"
            bbox = d.get('bbox', ('N/A',))
            bbox_str = ', '.join(str(int(v)) for v in bbox) if isinstance(bbox, (list, tuple)) else str(bbox)
            pdf.cell(10, 7, str(idx), border=1, align='C', fill=fill)
            pdf.cell(45, 7, dtype, border=1, align='C', fill=fill)
            pdf.cell(35, 7, conf, border=1, align='C', fill=fill)
            pdf.cell(80, 7, bbox_str, border=1, align='C', fill=fill, ln=1)
        pdf.ln(4)
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 8, 'No damage detected in this image.', 0, 1)
        pdf.ln(4)

    # ── 3  ANNOTATED IMAGE ──────────────────────────────────
    pdf.section_title('3.  Annotated Vehicle Image')

    img_to_embed = annotated_image
    if img_to_embed is None:
        # fallback: embed original
        img_to_embed = cv2.imread(image_path)

    if img_to_embed is not None:
        tmp = _save_temp_image(img_to_embed, 'annot_')
        temp_files.append(tmp)
        try:
            avail_w = 190 - pdf.l_margin
            pdf.image(tmp, x=pdf.l_margin, w=min(170, avail_w))
        except Exception:
            pdf.cell(0, 8, '[Image could not be embedded]', 0, 1)
        pdf.ln(4)

    # ── 4  DISCLAIMER ───────────────────────────────────────
    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        0,
        4,
        'Disclaimer: This report is auto-generated by an AI-powered vehicle inspection system. '
        'Damage types, severity, and cost estimates are indicative only and should be verified '
        'by a certified assessor before processing any insurance claim.'
    )

    # ── Save ────────────────────────────────────────────────
    pdf_filename = f"{claim_id}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    pdf.output(pdf_path)

    # cleanup temp images
    for t in temp_files:
        try:
            os.remove(t)
        except OSError:
            pass

    print(f"\n[PDF] Insurance Claim Report saved to: {pdf_path}")
    return os.path.abspath(pdf_path)


if __name__ == '__main__':
    # Quick standalone test with dummy data
    generate_claim_report(
        image_path='i.jpg',
        damages=[
            {'type': 'dent', 'confidence': 0.87, 'bbox': (120, 200, 310, 400)},
            {'type': 'scratch', 'confidence': 0.72, 'bbox': (350, 180, 500, 260)},
        ],
        plate_number='MH12AB1234',
    )
