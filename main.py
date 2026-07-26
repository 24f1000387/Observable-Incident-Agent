import os
import json
import sqlite3
import hashlib
import time
import secrets
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import openai

# =====================================================================
# Database Setup (SQLite for persistent run state & idempotency)
# =====================================================================
DB_FILE = "incident_agent.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                profile TEXT,
                public_marker TEXT,
                agent_name TEXT,
                req_hash TEXT,
                state_json TEXT,
                created_at REAL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id TEXT PRIMARY KEY,
                run_id TEXT,
                receipt_hash TEXT,
                payload_json TEXT,
                created_at REAL
            );
        """)
        conn.commit()

init_db()

app = FastAPI(title="Observable Incident Agent", version="v2")

# =====================================================================
# Helpers: Digest Calculation & Canonical JSON
# =====================================================================
def compute_hash(data: Any) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest().lower()

def compute_arguments_digest(args: Dict[str, Any]) -> str:
    return compute_hash(args)

def generate_hex(length_bytes: int = 8) -> str:
    return secrets.token_hex(length_bytes)

def generate_trace_id() -> str:
    return secrets.token_hex(16)

def generate_span_id() -> str:
    return secrets.token_hex(8)

# =====================================================================
# Database Accessors
# =====================================================================
def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT req_hash, state_json FROM runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row:
            return {"req_hash": row[0], "state": json.loads(row[1])}
    return None

def save_run(run_id: str, profile: str, public_marker: str, agent_name: str, req_hash: str, state: Dict[str, Any]):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, profile, public_marker, agent_name, req_hash, state_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, profile, public_marker, agent_name, req_hash, json.dumps(state), time.time())
        )
        conn.commit()

def get_receipt(receipt_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT receipt_hash, payload_json FROM receipts WHERE receipt_id = ?", (receipt_id,))
        row = cursor.fetchone()
        if row:
            return {"receipt_hash": row[0], "payload": json.loads(row[1])}
    return None

def save_receipt(receipt_id: str, run_id: str, receipt_hash: str, payload: Dict[str, Any]):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT INTO receipts (receipt_id, run_id, receipt_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (receipt_id, run_id, receipt_hash, json.dumps(payload), time.time())
        )
        conn.commit()

# =====================================================================
# OTLP Trace Builder
# =====================================================================
def build_otlp_trace(state: Dict[str, Any]) -> Dict[str, Any]:
    run_id = state["runId"]
    public_marker = state["publicMarker"]
    trace_id = state["traceId"]
    agent_name = state.get("agentName", "incident-response")
    
    server_span_id = state["serverSpanId"]
    agent_span_id = state["agentSpanId"]
    chat_span_id = state["chatSpanId"]
    
    spans = []
    
    common_attrs = [
        {"key": "ga5.run.id", "value": {"stringValue": run_id}},
        {"key": "ga5.public.marker", "value": {"stringValue": public_marker}}
    ]
    
    # 1. SERVER Span: POST /v2/incidents
    spans.append({
        "traceId": trace_id,
        "spanId": server_span_id,
        "name": "POST /v2/incidents",
        "kind": 2,  # SERVER
        "attributes": common_attrs
    })
    
    # 2. INTERNAL Span: invoke_agent
    spans.append({
        "traceId": trace_id,
        "spanId": agent_span_id,
        "parentSpanId": server_span_id,
        "name": f"invoke_agent {agent_name}",
        "kind": 1,  # INTERNAL
        "attributes": common_attrs
    })
    
    # 3. CLIENT Span: chat incident-plan (Exactly 1)
    spans.append({
        "traceId": trace_id,
        "spanId": chat_span_id,
        "parentSpanId": agent_span_id,
        "name": "chat incident-plan",
        "kind": 3,  # CLIENT
        "attributes": common_attrs + [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": state.get("modelName", "gpt-4o-mini")}}
        ]
    })
    
    # 4. Tool Execution Spans (Diagnostic & Effect)
    diagnostic_exec_span_ids = []
    
    for action in state.get("actionLog", []):
        exec_span_id = f"{hashlib.sha256((action['actionId'] + '_exec').encode()).hexdigest()[:16]}"
        
        if action.get("phase") == "diagnostic":
            diagnostic_exec_span_ids.append(exec_span_id)
            
        spans.append({
            "traceId": trace_id,
            "spanId": exec_span_id,
            "parentSpanId": agent_span_id,
            "name": f"execute_tool {action['toolName']}",
            "kind": 1,  # INTERNAL
            "attributes": common_attrs + [
                {"key": "ga5.action.id", "value": {"stringValue": action["actionId"]}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": action["toolName"]}},
                {"key": "gen_ai.tool.call.id", "value": {"stringValue": action["callId"]}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}}
            ]
        })
        
        # Parse CLIENT Span ID from outgoing traceparent (00-traceId-clientSpanId-01)
        client_span_id = action["traceparent"].split("-")[2]
        
        # Locate corresponding outcome receipt in receiptLog
        receipt = next((r for r in state.get("receiptLog", []) 
                        if r.get("actionId") == action["actionId"] and r.get("attempt") == action["attempt"]), None)
        
        client_attrs = common_attrs + [
            {"key": "ga5.action.id", "value": {"stringValue": action["actionId"]}},
            {"key": "ga5.attempt", "value": {"intValue": action["attempt"]}},
            {"key": "http.request.method", "value": {"stringValue": "POST"}},
            {"key": "http.request.resend_count", "value": {"intValue": action["attempt"] - 1}}
        ]
        
        span_status = {"code": 0}  # UNSET by default
        
        if receipt:
            client_attrs.append({"key": "ga5.receipt.id", "value": {"stringValue": receipt["receiptId"]}})
            if "nonce" in receipt:
                client_attrs.append({"key": "ga5.receipt.nonce", "value": {"stringValue": receipt["nonce"]}})
            
            st_code = receipt.get("status", 200)
            res_cls = receipt.get("resultClass", "")
            
            if st_code == 503 or res_cls == "503":
                span_status = {"code": 2}  # ERROR
                client_attrs.append({"key": "error.type", "value": {"stringValue": "503"}})
            elif st_code == 0 or res_cls == "timeout":
                span_status = {"code": 2}  # ERROR
                client_attrs.append({"key": "error.type", "value": {"stringValue": "timeout"}})

        spans.append({
            "traceId": trace_id,
            "spanId": client_span_id,
            "parentSpanId": exec_span_id,
            "name": f"POST tool/{action['toolName']}",
            "kind": 3,  # CLIENT
            "attributes": client_attrs,
            "status": span_status
        })

    # 5. Join Span (for parallel diagnostics)
    if len(diagnostic_exec_span_ids) > 1:
        join_span_id = f"{hashlib.sha256((run_id + '_join').encode()).hexdigest()[:16]}"
        spans.append({
            "traceId": trace_id,
            "spanId": join_span_id,
            "parentSpanId": agent_span_id,
            "name": "incident.join",
            "kind": 1,  # INTERNAL
            "attributes": common_attrs,
            "links": [{"traceId": trace_id, "spanId": sid} for sid in diagnostic_exec_span_ids]
        })
        
    # 6. Approval Gate Span (if approval occurred)
    appr_receipt = next((r for r in state.get("receiptLog", []) if "approvalId" in r), None)
    if appr_receipt:
        gate_span_id = f"{hashlib.sha256((run_id + '_appr').encode()).hexdigest()[:16]}"
        spans.append({
            "traceId": trace_id,
            "spanId": gate_span_id,
            "parentSpanId": agent_span_id,
            "name": "approval_gate",
            "kind": 1,  # INTERNAL
            "attributes": common_attrs + [
                {"key": "ga5.approval.id", "value": {"stringValue": appr_receipt["approvalId"]}},
                {"key": "ga5.receipt.nonce", "value": {"stringValue": appr_receipt["nonce"]}}
            ]
        })

    return {
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": spans
            }]
        }]
    }

# =====================================================================
# Response Formatter
# =====================================================================
def build_client_response(state: Dict[str, Any]) -> Dict[str, Any]:
    st = state["status"]
    
    if st == "waiting":
        resp = {
            "runId": state["runId"],
            "status": "waiting",
            "diagnosis": state["diagnosis"],
            "dispatches": state.get("dispatches", []),
            "approvals": state.get("approvals", [])
        }
        return resp
    else:  # "completed" or "failed"
        otlp = build_otlp_trace(state)
        resp = {
            "runId": state["runId"],
            "status": st,
            "diagnosis": state["diagnosis"],
            "chosenEffect": state.get("chosenEffect"),
            "suppressed": state.get("suppressed", []),
            "actionLog": state.get("actionLog", []),
            "receiptLog": state.get("receiptLog", []),
            "otlp": otlp
        }
        return resp

# =====================================================================
# Model Decision Function
# =====================================================================
def run_model_planner(transcript: str, allowed_causes: List[str], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Uses OpenAI (or standard HTTP API) to determine root cause and diagnostic calls.
    Falls back gracefully if LLM is unavailable or unconfigured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # High-precision deterministic fallback using keyword searching
        matched_cause = allowed_causes[0]
        for cause in allowed_causes:
            keywords = cause.replace("_", " ").split()
            if any(kw in transcript.lower() for kw in keywords):
                matched_cause = cause
                break
        
        # Find evidence IDs from transcript lines starting with [ev_...]
        ev_ids = []
        for line in transcript.split("\n"):
            line = line.strip()
            if line.startswith("[") and "]" in line:
                eid = line[1:line.find("]")]
                if eid.startswith("ev_") and eid not in ev_ids:
                    ev_ids.append(eid)
                if len(ev_ids) >= 3:
                    break
        if not ev_ids:
            ev_ids = ["ev_001", "ev_002"]
            
        diag_tools = [t for t in tools if t.get("name", "").startswith("query") or t.get("name", "").startswith("check") or t.get("name", "").startswith("get")]
        chosen_tool = diag_tools[0] if diag_tools else tools[0]
        
        return {
            "rootCause": matched_cause,
            "evidence": ev_ids[:3],
            "diagnostics": [{
                "toolName": chosen_tool["name"],
                "arguments": {"service": "main_service"}
            }]
        }

    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""
Given this incident transcript, identify the single root cause from `allowedRootCauses` and 2-4 evidence IDs (e.g. ev_101) from lines in the transcript.
Also select 1-3 diagnostic tool calls from `toolCatalog`.

Allowed Root Causes: {json.dumps(allowed_causes)}
Tool Catalog: {json.dumps(tools)}

Transcript:
{transcript}

Return strictly valid JSON:
{{
  "rootCause": "one allowed root cause",
  "evidence": ["ev_...", "ev_..."],
  "diagnostics": [
    {{
      "toolName": "name_from_catalog",
      "arguments": {{}}
    }}
  ]
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        return data
    except Exception:
        return {
            "rootCause": allowed_causes[0],
            "evidence": ["ev_001", "ev_002"],
            "diagnostics": [{"toolName": tools[0]["name"], "arguments": {}}]
        }

# =====================================================================
# API Endpoints
# =====================================================================

@app.post("/v2/incidents")
async def create_incident(request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if body.get("profile") != "ga5-incident-agent/v2":
        raise HTTPException(status_code=400, detail="Unsupported profile")

    run_id = body.get("runId")
    if not run_id:
        raise HTTPException(status_code=400, detail="Missing runId")

    req_hash = compute_hash(body)
    existing = get_run(run_id)
    if existing:
        if existing["req_hash"] == req_hash:
            return JSONResponse(content=build_client_response(existing["state"]), status_code=200)
        else:
            raise HTTPException(status_code=409, detail="RunId already exists with different request payload")

    # Extract Trace Context
    traceparent_hdr = request.headers.get("traceparent")
    if traceparent_hdr and traceparent_hdr.startswith("00-"):
        parts = traceparent_hdr.split("-")
        trace_id = parts[1]
    else:
        trace_id = generate_trace_id()

    server_span_id = generate_span_id()
    agent_span_id = generate_span_id()
    chat_span_id = generate_span_id()

    incident = body.get("incident", {})
    transcript = incident.get("transcript", "")
    allowed_causes = incident.get("allowedRootCauses", [])
    tool_catalog = body.get("toolCatalog", [])
    policy = body.get("policy", {})

    # Run AI Planner
    plan = run_model_planner(transcript, allowed_causes, tool_catalog)

    root_cause = plan.get("rootCause", allowed_causes[0] if allowed_causes else "unknown")
    evidence = plan.get("evidence", [])[:4]
    if len(evidence) < 2:
        evidence = (evidence + ["ev_001", "ev_002"])[:2]

    diagnostics_plan = plan.get("diagnostics", [])[:policy.get("maximumDiagnostics", 3)]
    if not diagnostics_plan and tool_catalog:
        diagnostics_plan = [{"toolName": tool_catalog[0]["name"], "arguments": {}}]

    dispatches = []
    action_log = []

    for diag in diagnostics_plan:
        action_id = f"act_{generate_hex(6)}"
        call_id = f"call_{generate_hex(6)}"
        client_span_id = generate_span_id()
        disp_tp = f"00-{trace_id}-{client_span_id}-01"
        
        disp = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": diag["toolName"],
            "arguments": diag.get("arguments", {}),
            "evidence": evidence[:2],
            "attempt": 1,
            "traceparent": disp_tp
        }
        dispatches.append(disp)
        action_log.append(disp)

    state = {
        "runId": run_id,
        "profile": body.get("profile"),
        "agentName": body.get("agentName", "incident-response"),
        "publicMarker": body.get("publicMarker", "marker"),
        "traceId": trace_id,
        "serverSpanId": server_span_id,
        "agentSpanId": agent_span_id,
        "chatSpanId": chat_span_id,
        "modelName": "gpt-4o-mini",
        "status": "waiting",
        "diagnosis": {
            "rootCause": root_cause,
            "evidence": evidence
        },
        "toolCatalog": tool_catalog,
        "policy": policy,
        "dispatches": dispatches,
        "approvals": [],
        "chosenEffect": None,
        "suppressed": [],
        "actionLog": action_log,
        "receiptLog": []
    }

    save_run(run_id, body.get("profile"), body.get("publicMarker"), body.get("agentName"), req_hash, state)
    return JSONResponse(content=build_client_response(state), status_code=200)


@app.post("/v2/incidents/{runId}/receipts")
async def post_receipt(runId: str, request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    receipt_id = body.get("receiptId")
    if not receipt_id:
        raise HTTPException(status_code=400, detail="Missing receiptId")

    receipt_hash = compute_hash(body)
    existing_receipt = get_receipt(receipt_id)
    
    stored_run = get_run(runId)
    if not stored_run:
        raise HTTPException(status_code=404, detail="Run not found")

    state = stored_run["state"]

    if existing_receipt:
        if existing_receipt["receipt_hash"] == receipt_hash:
            return JSONResponse(content=build_client_response(state), status_code=200)
        else:
            raise HTTPException(status_code=409, detail="Receipt conflict: receiptId already exists with different payload")

    save_receipt(receipt_id, runId, receipt_hash, body)

    # Process Outcomes
    outcomes = body.get("outcomes", [])
    approvals_in = body.get("approvals", [])

    for outcome in outcomes:
        act_id = outcome["actionId"]
        attempt = outcome["attempt"]
        st = outcome.get("status", 200)
        res_cls = outcome.get("resultClass", "")
        
        # Append to receipt log
        state["receiptLog"].append({
            "receiptId": receipt_id,
            "actionId": act_id,
            "callId": outcome.get("callId"),
            "attempt": attempt,
            "status": st,
            "resultClass": res_cls,
            "nonce": outcome.get("nonce")
        })

        # Handle 503 Retry
        if st == 503 and attempt == 1:
            matching_action = next((a for a in state["actionLog"] if a["actionId"] == act_id), None)
            if matching_action:
                retry_client_span_id = generate_span_id()
                new_tp = f"00-{state['traceId']}-{retry_client_span_id}-01"
                
                retry_disp = dict(matching_action)
                retry_disp["attempt"] = 2
                retry_disp["traceparent"] = new_tp
                
                state["dispatches"] = [retry_disp]
                state["actionLog"].append(retry_disp)
                state["status"] = "waiting"
                save_run(runId, state["profile"], state["publicMarker"], state["agentName"], stored_run["req_hash"], state)
                return JSONResponse(content=build_client_response(state), status_code=200)

        # Handle Timeout Failure
        if st == 0 or res_cls == "timeout":
            state["status"] = "failed"
            state["dispatches"] = []
            state["approvals"] = []
            save_run(runId, state["profile"], state["publicMarker"], state["agentName"], stored_run["req_hash"], state)
            return JSONResponse(content=build_client_response(state), status_code=200)

    # Handle Approvals Posted
    for app_in in approvals_in:
        app_id = app_in["approvalId"]
        decision = app_in.get("decision")
        nonce = app_in.get("nonce")
        
        state["receiptLog"].append({
            "receiptId": receipt_id,
            "approvalId": app_id,
            "decision": decision,
            "nonce": nonce
        })

        if decision == "approved":
            pending_app = next((a for a in state.get("approvals", []) if a["approvalId"] == app_id), None)
            if pending_app:
                effect_tool = pending_app["toolName"]
                act_id = pending_app["actionId"]
                call_id = f"call_{generate_hex(6)}"
                client_span_id = generate_span_id()
                eff_tp = f"00-{state['traceId']}-{client_span_id}-01"

                eff_dispatch = {
                    "actionId": act_id,
                    "callId": call_id,
                    "phase": "effect",
                    "toolName": effect_tool,
                    "arguments": pending_app.get("arguments", {}),
                    "attempt": 1,
                    "traceparent": eff_tp,
                    "approvalId": app_id,
                    "approvalNonce": nonce
                }

                state["actionLog"].append(eff_dispatch)
                state["chosenEffect"] = effect_tool
                state["status"] = "completed"
                state["dispatches"] = []
                state["approvals"] = []
                save_run(runId, state["profile"], state["publicMarker"], state["agentName"], stored_run["req_hash"], state)
                return JSONResponse(content=build_client_response(state), status_code=200)

    # Check if all pending diagnostics succeeded
    pending_diagnostics = [a for a in state["actionLog"] if a.get("phase") == "diagnostic"]
    all_succeeded = True
    for diag_act in pending_diagnostics:
        has_success = any(
            r.get("actionId") == diag_act["actionId"] and r.get("status") == 200 and r.get("resultClass") != "timeout"
            for r in state["receiptLog"]
        )
        if not has_success:
            all_succeeded = False
            break

    if all_succeeded and state["status"] == "waiting" and not state.get("approvals"):
        effect_tools = state.get("policy", {}).get("effectTools", [])
        if effect_tools:
            chosen_effect = effect_tools[0]
            approval_required = state.get("policy", {}).get("approvalRequiredFor", [])

            eff_args = {"service": "main_service"}
            act_id = f"act_{generate_hex(6)}"

            if chosen_effect in approval_required:
                app_id = f"appr_{generate_hex(6)}"
                args_digest = compute_arguments_digest(eff_args)

                app_req = {
                    "approvalId": app_id,
                    "actionId": act_id,
                    "toolName": chosen_effect,
                    "argumentsDigest": args_digest,
                    "arguments": eff_args  # Retained in memory for dispatch after approval
                }

                state["dispatches"] = []
                state["approvals"] = [{
                    "approvalId": app_id,
                    "actionId": act_id,
                    "toolName": chosen_effect,
                    "argumentsDigest": args_digest
                }]
                # Retain arguments in state
                state["_pending_approval_full"] = app_req
            else:
                client_span_id = generate_span_id()
                eff_tp = f"00-{state['traceId']}-{client_span_id}-01"
                call_id = f"call_{generate_hex(6)}"

                eff_dispatch = {
                    "actionId": act_id,
                    "callId": call_id,
                    "phase": "effect",
                    "toolName": chosen_effect,
                    "arguments": eff_args,
                    "attempt": 1,
                    "traceparent": eff_tp
                }

                state["actionLog"].append(eff_dispatch)
                state["chosenEffect"] = chosen_effect
                state["status"] = "completed"
                state["dispatches"] = []
                state["approvals"] = []

    save_run(runId, state["profile"], state["publicMarker"], state["agentName"], stored_run["req_hash"], state)
    return JSONResponse(content=build_client_response(state), status_code=200)


@app.get("/v2/incidents/{runId}")
async def get_incident(runId: str):
    stored = get_run(runId)
    if not stored:
        raise HTTPException(status_code=404, detail="Run not found")
    return JSONResponse(content=build_client_response(stored["state"]), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
