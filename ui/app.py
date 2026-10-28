"""Local web UI for the retail-banking-chatbot Mistral Workflow.

Two pages:
  - /      : basic chat UI driving the deployed workflow via the Mistral Workflows API
  - /logs  : minimal log/trace viewer backed by a local SQLite store

The app talks to the SAME managed deployment used by Vibe Work and Studio:
it starts executions, reads pending inputs, submits user messages, and
captures logs / events / the OpenTelemetry trace for each execution.

Run (from the repo root):
    export MISTRAL_API_KEY=...
    uvicorn ui.app:app --reload --port 8000

Then open http://localhost:8000 (chat) and http://localhost:8000/logs (logs).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mistralai import Mistral

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
WORKFLOW_NAME = os.environ.get("WORKFLOW_NAME", "retail-banking-chatbot")
DB_PATH = os.environ.get("UI_DB_PATH", os.path.join(os.path.dirname(__file__), "data", "ui.db"))
POLL_SECONDS = float(os.environ.get("UI_POLL_SECONDS", "1.5"))

_client: Optional[Mistral] = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
_sync_lock = threading.Lock()  # one sync per execution at a time

app = FastAPI(title="Retail Banking Chatbot — local UI", version="0.1.0")

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _require_client() -> Mistral:
    if _client is None:
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY is not set. Export it before starting the UI.",
        )
    return _client


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------
@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db() -> None:
    with _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                execution_id TEXT PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                status TEXT,
                last_event_id TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                event_id TEXT UNIQUE,
                event_type TEXT,
                ts_ns TEXT,
                run_id TEXT,
                task_type TEXT,
                text TEXT,
                payload_json TEXT
            );
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                severity TEXT,
                body TEXT,
                UNIQUE(execution_id, ts, body)
            );
            CREATE TABLE IF NOT EXISTS spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                span_id TEXT,
                parent_span_id TEXT,
                name TEXT,
                start_time_unix_nano TEXT,
                duration_ms REAL,
                attributes_json TEXT,
                UNIQUE(execution_id, span_id)
            );
            """
        )


_init_db()


# ---------------------------------------------------------------------------
# Generic helpers over SDK responses (defensive: shapes vary across endpoints)
# ---------------------------------------------------------------------------
def _unwrap_payload(value: Any) -> Any:
    """SDK payloads may arrive as {'type': 'json', 'value': ...} or decoded."""
    if isinstance(value, dict) and set(value.keys()) <= {"type", "value", "encoding_options"} and "value" in value:
        return _unwrap_payload(value["value"])
    return value


def _to_dict(obj: Any) -> Any:
    """Recursively convert SDK models / TypedDicts to plain JSON-able values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):  # pydantic model
        try:
            return _to_dict(obj.model_dump(mode="json"))
        except TypeError:
            return _to_dict(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _to_dict(vars(obj))
    return str(obj)


def _iter_event_dicts(node: Any) -> Iterator[Dict[str, Any]]:
    """Walk any JSON-ish structure and yield dicts that look like workflow events."""
    if isinstance(node, dict):
        if "event_type" in node or ("attributes" in node and isinstance(node["attributes"], dict)):
            yield node
        for v in node.values():
            yield from _iter_event_dicts(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_event_dicts(v)


def _event_fields(event: Dict[str, Any]) -> Dict[str, Any]:
    attrs = event.get("attributes") or event
    out = {
        "event_id": event.get("event_id") or attrs.get("event_id"),
        "event_type": event.get("event_type") or attrs.get("event_type"),
        "ts_ns": event.get("event_timestamp_ns") or attrs.get("event_timestamp_ns"),
        "run_id": event.get("run_id") or attrs.get("run_id"),
    }
    out["task_type"] = attrs.get("custom_task_type")
    payload = _unwrap_payload(attrs.get("payload"))
    out["payload"] = _to_dict(payload) if payload is not None else None
    return out


def _assistant_text(payload: Any) -> Optional[str]:
    """Extract the text of an assistant_message payload."""
    if not isinstance(payload, dict):
        return None
    chunks = payload.get("contentChunks") or payload.get("content") or []
    if isinstance(chunks, list):
        texts = [
            c.get("text", "")
            for c in chunks
            if isinstance(c, dict) and c.get("type", "text") == "text"
        ]
        if texts:
            return "\n".join(t for t in texts if t)
    return None


def _prompt_info(payload: Any) -> Optional[Dict[str, Any]]:
    """Extract label/suggestions from a wait_for_input payload."""
    if not isinstance(payload, dict):
        return None
    schema = payload.get("input_schema") or {}
    if not isinstance(schema, dict):
        return None
    message_prop = (schema.get("properties") or {}).get("message") or {}
    examples = message_prop.get("examples") or []
    suggestions = []
    for ex in examples:
        if isinstance(ex, list):
            suggestions.append("".join(c.get("text", "") for c in ex if isinstance(c, dict)))
    return {
        "label": payload.get("label") or message_prop.get("description"),
        "suggestions": suggestions,
    }


# ---------------------------------------------------------------------------
# Sync: pull history events, logs and the OTel trace into SQLite
# ---------------------------------------------------------------------------
def _sync_execution(execution_id: str) -> Dict[str, Any]:
    """Fetch new data for one execution from the Mistral API into SQLite.

    Returns the chat-visible state: new assistant messages, pending prompt,
    execution status and final result.
    """
    if not _sync_lock.acquire(blocking=False):
        return {"busy": True, "messages": [], "pending": None}
    try:
        client = _require_client()
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Unknown execution_id")

        # --- 1. history events (assistant messages, wait_for_input, lifecycle)
        new_messages: List[Dict[str, Any]] = []
        last_event_id: Optional[str] = None
        try:
            history = client.workflows.executions.get_workflow_execution_history(
                execution_id=execution_id, decode_payloads=True
            )
            history = _to_dict(_unwrap_payload(history))
        except Exception:
            history = None
        if history:
            seen_last = row["last_event_id"]
            passed_seen = seen_last is None
            for ev in _iter_event_dicts(history):
                f = _event_fields(ev)
                if not f["event_type"]:
                    continue
                if not passed_seen:
                    if f["event_id"] == seen_last:
                        passed_seen = True
                    continue
                last_event_id = f["event_id"] or last_event_id
                if f["event_type"] not in ("CUSTOM_TASK_STARTED", "CUSTOM_TASK_COMPLETED"):
                    continue
                if f["task_type"] == "assistant_message" and f["event_type"] == "CUSTOM_TASK_STARTED":
                    text = _assistant_text(f["payload"])
                    if text:
                        new_messages.append({"role": "assistant", "text": text})
                with _db() as conn:
                    conn.execute(
                        """INSERT OR IGNORE INTO events
                           (execution_id, event_id, event_type, ts_ns, run_id, task_type, text, payload_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            execution_id,
                            f["event_id"] or f"{execution_id}:{f['ts_ns']}:{f['event_type']}",
                            f["event_type"],
                            f["ts_ns"],
                            f["run_id"],
                            f["task_type"],
                            _assistant_text(f["payload"]),
                            json.dumps(f["payload"], default=str) if f["payload"] else None,
                        ),
                    )

        # --- 2. pending input (what the workflow is waiting for right now)
        pending: Optional[Dict[str, Any]] = None
        try:
            q = client.workflows.executions.query_workflow_execution(
                execution_id=execution_id,
                name="__get_pending_inputs",
                input={},
            )
            q = _to_dict(q)
            items = (q or {}).get("result") or (q or {}).get("pending_inputs") or []
            if isinstance(q, dict) and "pending_inputs" not in q and isinstance(q.get("result"), dict):
                items = q["result"].get("pending_inputs", [])
            if isinstance(items, list) and items:
                first = items[0]
                payload = {"label": first.get("label"), "input_schema": first.get("input_schema")}
                info = _prompt_info(payload) or {}
                pending = {
                    "task_id": first.get("task_id"),
                    "label": info.get("label"),
                    "suggestions": info.get("suggestions") or [],
                }
        except Exception:
            pending = None

        # --- 3. execution status / result
        status: Optional[str] = None
        result_text: Optional[str] = None
        try:
            ex = _to_dict(client.workflows.executions.get_workflow_execution(execution_id=execution_id))
            status = ex.get("status")
            result = ex.get("result")
            if isinstance(result, dict):
                content = result.get("content") or []
                texts = []
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        texts.append(c["text"])
                result_text = "\n".join(texts) or None
        except Exception:
            pass

        # --- 4. execution logs -> SQLite
        try:
            logs_resp = client.workflows.executions.get_workflow_execution_logs(
                execution_id=execution_id, limit=100, order="asc"
            )
            logs_resp = _to_dict(logs_resp)
            for lg in (logs_resp or {}).get("logs") or []:
                if not isinstance(lg, dict):
                    continue
                with _db() as conn:
                    conn.execute(
                        """INSERT OR IGNORE INTO logs (execution_id, ts, severity, body)
                           VALUES (?, ?, ?, ?)""",
                        (
                            execution_id,
                            lg.get("timestamp") or "",
                            lg.get("severity") or "INFO",
                            lg.get("body") or "",
                        ),
                    )
        except Exception:
            pass

        # --- 5. OTel trace -> SQLite spans
        try:
            trace = _to_dict(
                client.workflows.executions.get_workflow_execution_trace_otel(
                    execution_id=execution_id
                )
            )
            for span in _iter_spans(trace):
                with _db() as conn:
                    conn.execute(
                        """INSERT OR IGNORE INTO spans
                           (execution_id, span_id, parent_span_id, name,
                            start_time_unix_nano, duration_ms, attributes_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            execution_id,
                            span.get("spanID") or span.get("span_id"),
                            span.get("parentSpanID") or span.get("parent_span_id"),
                            span.get("name"),
                            str(span.get("startTimeUnixNano") or span.get("start_time_unix_nano") or ""),
                            _duration_ms(span),
                            json.dumps(
                                {"attributes": span.get("attributes"), "kind": span.get("kind")},
                                default=str,
                            ),
                        ),
                    )
        except Exception:
            pass

        with _db() as conn:
            conn.execute(
                """UPDATE sessions SET status = ?, last_event_id = COALESCE(?, last_event_id),
                   updated_at = ? WHERE execution_id = ?""",
                (
                    status or row["status"],
                    last_event_id,
                    datetime.now(timezone.utc).isoformat(),
                    execution_id,
                ),
            )

        return {
            "busy": False,
            "execution_id": execution_id,
            "status": status or row["status"],
            "messages": new_messages,
            "pending": pending,
            "result": result_text,
        }
    finally:
        _sync_lock.release()


def _iter_spans(trace: Any) -> Iterator[Dict[str, Any]]:
    """Extract OTLP span dicts from a Tempo GetTraceResponse-like structure."""
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "spanID" in node or "span_id" in node:
                yield_from_spans.append(node)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    yield_from_spans: List[Dict[str, Any]] = []
    walk(_to_dict(trace))
    yield from yield_from_spans


def _duration_ms(span: Dict[str, Any]) -> Optional[float]:
    d = span.get("durationNanos") or span.get("duration_nanos")
    if d is not None:
        try:
            return float(d) / 1e6
        except (TypeError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
class StartSessionBody(BaseModel):
    user_id: str = "demo"


class MessageBody(BaseModel):
    execution_id: str
    text: str


@app.post("/api/session")
def start_session(body: StartSessionBody) -> Dict[str, Any]:
    client = _require_client()
    execution = client.workflows.execute_workflow(
        workflow_identifier=WORKFLOW_NAME,
        input={"user_id": body.user_id},
    )
    ex = _to_dict(execution)
    execution_id = ex.get("execution_id") or ex.get("id")
    if not execution_id:
        raise HTTPException(status_code=500, detail="No execution_id returned by the API")
    with _db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (execution_id, workflow_name, user_id, started_at, status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                execution_id,
                ex.get("workflow_name") or WORKFLOW_NAME,
                body.user_id,
                datetime.now(timezone.utc).isoformat(),
                ex.get("status") or "RUNNING",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    _sync_execution(execution_id)
    return {"execution_id": execution_id, "status": ex.get("status") or "RUNNING"}


@app.post("/api/message")
def send_message(body: MessageBody) -> Dict[str, Any]:
    client = _require_client()
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    # Resolve the pending input task the workflow is waiting on.
    q = _to_dict(
        client.workflows.executions.query_workflow_execution(
            execution_id=body.execution_id,
            name="__get_pending_inputs",
            input={},
        )
    )
    items = (q or {}).get("result") or (q or {}).get("pending_inputs") or []
    if isinstance(q, dict) and isinstance(q.get("result"), dict):
        items = q["result"].get("pending_inputs", [])
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=409, detail="The workflow is not waiting for any input")

    task_id = items[0].get("task_id")
    update_input = {
        "task_id": task_id,
        "input": {"message": [{"type": "text", "text": text}]},
    }
    client.workflows.executions.update_workflow_execution(
        execution_id=body.execution_id,
        name="__submit_input",
        input=update_input,
    )
    return {"ok": True}


@app.get("/api/updates")
def updates(execution_id: str) -> Dict[str, Any]:
    return _sync_execution(execution_id)


@app.get("/api/executions")
def list_executions() -> Dict[str, Any]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT s.execution_id, s.user_id, s.started_at, s.status, s.updated_at,
                      (SELECT COUNT(*) FROM events e WHERE e.execution_id = s.execution_id) AS events,
                      (SELECT COUNT(*) FROM logs l WHERE l.execution_id = s.execution_id) AS logs,
                      (SELECT COUNT(*) FROM spans p WHERE p.execution_id = s.execution_id) AS spans
               FROM sessions s ORDER BY s.started_at DESC LIMIT 100"""
        ).fetchall()
    return {"executions": [dict(r) for r in rows]}


@app.get("/api/logs")
def get_logs(execution_id: str, limit: int = 500) -> Dict[str, Any]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT ts, severity, body FROM logs WHERE execution_id = ? ORDER BY ts ASC LIMIT ?",
            (execution_id, limit),
        ).fetchall()
    return {"logs": [dict(r) for r in rows]}


@app.get("/api/events")
def get_events(execution_id: str, limit: int = 500) -> Dict[str, Any]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT event_id, event_type, ts_ns, run_id, task_type, text
               FROM events WHERE execution_id = ? ORDER BY id ASC LIMIT ?""",
            (execution_id, limit),
        ).fetchall()
    return {"events": [dict(r) for r in rows]}


@app.get("/api/trace")
def get_trace(execution_id: str) -> Dict[str, Any]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT span_id, parent_span_id, name, start_time_unix_nano, duration_ms, attributes_json
               FROM spans WHERE execution_id = ? ORDER BY start_time_unix_nano ASC""",
            (execution_id,),
        ).fetchall()
    spans = []
    for r in rows:
        d = dict(r)
        try:
            d["attributes"] = json.loads(d.pop("attributes_json") or "{}")
        except json.JSONDecodeError:
            d["attributes"] = {}
        spans.append(d)
    return {"spans": spans}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def chat_page() -> HTMLResponse:
    with open(os.path.join(_STATIC_DIR, "chat.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/logs", response_class=HTMLResponse)
def logs_page() -> HTMLResponse:
    with open(os.path.join(_STATIC_DIR, "logs.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())
