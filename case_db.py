import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class CaseRow:
    case_id: str
    created_at: str
    status: str
    plate: str
    entry_image: str
    exit_image: str
    entry_severity: int
    exit_severity: int
    result: str
    report_pdf_name: str


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                plate TEXT NOT NULL,
                entry_image TEXT NOT NULL,
                exit_image TEXT NOT NULL,
                entry_severity INTEGER NOT NULL,
                exit_severity INTEGER NOT NULL,
                result TEXT NOT NULL,
                report_pdf_name TEXT NOT NULL,
                entry_damage_image TEXT NOT NULL DEFAULT '',
                exit_damage_image TEXT NOT NULL DEFAULT '',
                exit_anpr_image TEXT NOT NULL DEFAULT '',
                ai_report TEXT NOT NULL DEFAULT '',
                duration_s REAL NOT NULL DEFAULT 0,
                plate_crop TEXT NOT NULL DEFAULT ''
            )
            """
        )

        cur = conn.execute("PRAGMA table_info(cases)")
        existing = {r[1] for r in cur.fetchall()}
        desired = {
            ("entry_damage_image", "TEXT NOT NULL DEFAULT ''"),
            ("exit_damage_image", "TEXT NOT NULL DEFAULT ''"),
            ("exit_anpr_image", "TEXT NOT NULL DEFAULT ''"),
            ("ai_report", "TEXT NOT NULL DEFAULT ''"),
            ("duration_s", "REAL NOT NULL DEFAULT 0"),
            ("plate_crop", "TEXT NOT NULL DEFAULT ''"),
        }
        for col, ddl in desired:
            if col not in existing:
                conn.execute(f"ALTER TABLE cases ADD COLUMN {col} {ddl}")
        conn.commit()
    finally:
        conn.close()


def upsert_case(db_path: str, data: Dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO cases (
                case_id, created_at, status, plate, entry_image, exit_image,
                entry_severity, exit_severity, result, report_pdf_name
                , entry_damage_image, exit_damage_image, exit_anpr_image, ai_report, duration_s, plate_crop
            ) VALUES (
                :case_id, :created_at, :status, :plate, :entry_image, :exit_image,
                :entry_severity, :exit_severity, :result, :report_pdf_name,
                :entry_damage_image, :exit_damage_image, :exit_anpr_image, :ai_report, :duration_s, :plate_crop
            )
            ON CONFLICT(case_id) DO UPDATE SET
                status=excluded.status,
                plate=excluded.plate,
                entry_image=excluded.entry_image,
                exit_image=excluded.exit_image,
                entry_severity=excluded.entry_severity,
                exit_severity=excluded.exit_severity,
                result=excluded.result,
                report_pdf_name=excluded.report_pdf_name,
                entry_damage_image=excluded.entry_damage_image,
                exit_damage_image=excluded.exit_damage_image,
                exit_anpr_image=excluded.exit_anpr_image,
                ai_report=excluded.ai_report,
                duration_s=excluded.duration_s,
                plate_crop=excluded.plate_crop
            """
            ,
            data,
        )
        conn.commit()
    finally:
        conn.close()


def list_cases(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM cases ORDER BY datetime(created_at) DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_case(db_path: str, case_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        cur = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
