from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import threading
import time
import uuid

import case_db

from pipeline_wrapper import run_full_analysis

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

DB_PATH = "cases.db"
case_db.init_db(DB_PATH)

JOBS = {}
JOBS_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _make_steps():
    labels = [
        "Plate detect",
        "Enhance",
        "OCR",
        "Normalize",
        "Features",
        "Fusion",
        "Report",
    ]
    return [{"label": l, "state": "idle"} for l in labels]


def _set_step(steps, label, state):
    for s in steps:
        if s.get("label") == label:
            s["state"] = state
            return


def _job_set(job_id: str, patch: dict) -> None:
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            return
        j.update(patch)


def _job_get(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        return dict(j) if j else None


def _save_upload(file: UploadFile, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


def _calc_result(severity: int) -> str:
    return "Damage" if severity >= 8 else "Cleared"


def _run_case_job(job_id: str) -> None:
    started = _now()
    job = _job_get(job_id)
    if not job:
        return

    steps = job.get("steps") or _make_steps()
    _job_set(job_id, {"status": "running", "steps": steps, "started_at": started})

    case_id = job["case_id"]
    image_path = job["image"]

    try:
        _set_step(steps, "Enhance", "running")
        _job_set(job_id, {"steps": steps})

        out = run_full_analysis(
            image_path,
            output_prefix=f"{case_id}_img",
            generate_pdf=True,
            generate_ai=True,
        )

        _set_step(steps, "Enhance", "done")
        _set_step(steps, "Plate detect", "done")
        _set_step(steps, "OCR", "done")
        _set_step(steps, "Normalize", "done")
        _set_step(steps, "Features", "done")
        _set_step(steps, "Fusion", "done")
        _set_step(steps, "Report", "done")

        severity = int(out.get("severity", 0))
        plate = out.get("plate", "Unknown")
        damage_image = out.get("damage_image", "")
        anpr_image = out.get("anpr_image", "")
        ai_report = out.get("ai_report", "")
        pdf_name = out.get("report_pdf_name", "")

        result = _calc_result(severity)

        duration = _now() - started

        case_db.upsert_case(
            DB_PATH,
            {
                "case_id": case_id,
                "created_at": job["created_at"],
                "status": "done",
                "plate": plate,
                "entry_image": "",
                "exit_image": image_path.replace('\\', '/'),
                "entry_severity": 0,
                "exit_severity": severity,
                "result": result,
                "report_pdf_name": pdf_name,
                "entry_damage_image": "",
                "exit_damage_image": damage_image,
                "exit_anpr_image": anpr_image,
                "ai_report": ai_report,
                "duration_s": float(duration),
                "plate_crop": out.get("plate_crop", ""),
            },
        )

        _job_set(
            job_id,
            {
                "status": "done",
                "runtime_s": duration,
                "plate": plate,
                "result": result,
                "report_pdf_name": pdf_name,
                "plate_crop": out.get("plate_crop", ""),
                "steps": steps,
                "outputs": {
                    "entry_damage_image": "",
                    "exit_damage_image": damage_image,
                    "exit_anpr_image": anpr_image,
                },
            },
        )
    except Exception as e:
        duration = _now() - started
        _job_set(job_id, {"status": "error", "runtime_s": duration, "error": str(e), "steps": steps})


@app.get("/app", response_class=HTMLResponse)
async def app_home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/api/start")
async def api_start(
    request: Request,
    image: UploadFile | None = File(default=None),
    demo: str | None = Form(default=None),
):
    case_id = f"CASE-{uuid.uuid4().hex[:10].upper()}"
    job_id = uuid.uuid4().hex

    created_at = case_db.now_iso()

    os.makedirs("uploads", exist_ok=True)

    if demo:
        image_path = os.path.join("uploads", f"{case_id}_img.jpg")
        if not os.path.exists("i.jpg"):
            raise HTTPException(status_code=404, detail="demo_image_missing")
        shutil.copyfile("i.jpg", image_path)
    else:
        if image is None:
            raise HTTPException(status_code=400, detail="image_required")
        image_path = os.path.join("uploads", f"{case_id}_img_{image.filename}")
        _save_upload(image, image_path)

    case_db.upsert_case(
        DB_PATH,
        {
            "case_id": case_id,
            "created_at": created_at,
            "status": "running",
            "plate": "Processing",
            "entry_image": "",
            "exit_image": image_path.replace('\\', '/'),
            "entry_severity": 0,
            "exit_severity": 0,
            "result": "Processing",
            "report_pdf_name": "",
            "entry_damage_image": "",
            "exit_damage_image": "",
            "exit_anpr_image": "",
            "ai_report": "",
            "duration_s": 0.0,
            "plate_crop": "",
        },
    )

    job = {
        "job_id": job_id,
        "case_id": case_id,
        "created_at": created_at,
        "status": "queued",
        "steps": _make_steps(),
        "image": image_path.replace('\\', '/'),
        "plate": "—",
        "result": "—",
        "runtime_s": None,
        "error": None,
        "outputs": {},
    }

    with JOBS_LOCK:
        JOBS[job_id] = job

    t = threading.Thread(target=_run_case_job, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id, "case_id": case_id}


@app.get("/api/job/{job_id}")
async def api_job(job_id: str):
    j = _job_get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job_not_found")
    return JSONResponse(j)


@app.get("/api/history")
async def api_history():
    rows = case_db.list_cases(DB_PATH, limit=100)
    out = []
    for r in rows:
        out.append(
            {
                "case_id": r.get("case_id"),
                "created_at": r.get("created_at"),
                "status": r.get("status"),
                "plate": r.get("plate"),
                "plate_crop": r.get("plate_crop"),
                "entry_severity": r.get("entry_severity"),
                "exit_severity": r.get("exit_severity"),
                "result": r.get("result"),
                "duration_s": r.get("duration_s"),
            }
        )
    return JSONResponse(out)


@app.get("/api/case/{case_id}")
async def api_case(case_id: str):
    r = case_db.get_case(DB_PATH, case_id)
    if not r:
        raise HTTPException(status_code=404, detail="case_not_found")
    return JSONResponse(r)


@app.get("/report/view/{pdf_name}")
async def view_report(pdf_name: str):
    pdf_path = os.path.join("outputs", pdf_name)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="report_not_found")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{pdf_name}\""},
    )


@app.get("/report/download/{pdf_name}")
async def download_report(pdf_name: str):
    pdf_path = os.path.join("outputs", pdf_name)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="report_not_found")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_name,
        headers={"Content-Disposition": f"attachment; filename=\"{pdf_name}\""},
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo(request: Request):
    src = "i.jpg"
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="demo_image_not_found")

    os.makedirs("uploads", exist_ok=True)
    demo_name = "demo_i.jpg"
    dst = os.path.join("uploads", demo_name)
    shutil.copyfile(src, dst)

    result = run_full_analysis(dst)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "image": dst.replace('\\\\', '/')
        }
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    path = f"uploads/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_full_analysis(path)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "image": path
        }
    )