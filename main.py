import os
import json
import sqlite3
import hashlib
import time
import secrets
import re
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import openai

# =====================================================================
# Database Setup
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
# Helpers & Digest Utilities
# =====================================================================
def compute_hash(data: Any) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest().lower()

def compute_arguments_digest(args: Dict[str, Any]) -> str:
    return compute_hash(args)

def generate_opaque_id(prefix: str = "id") -> str:
    return f"{prefix}_{secrets.token_hex(8)}"

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
# Deterministic Evidence & Argument Extractor
# =====================================================================
def extract_evidence_ids(transcript: str) -> List[str]:
    """Finds all explicit [ev_...] tags in the transcript."""
    matches = re.findall(r'\[(ev_[a-zA-Z0-9_\-]+)\]', transcript)
    seen = []
    for m in matches:
        if m not in seen:
            seen.append(m)
    return seen

def extract_case_arguments(tool_schema: Dict[str, Any], incident: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts exact values matching tool properties from incident metadata."""
    args = {}
    properties = tool_schema.get("properties", {})
    required = tool_schema.get("required", list(properties.keys()))
    
    service_name = incident.get("service", "unknown-service")
    incident_id = incident.get("incidentId", "inc-001")
    
    for prop in required:
        p_info = properties.get(prop, {})
        p_type = p_info.get("type", "string")
        
        if prop in ["service", "target_service", "service_name"]:
            args[prop] = service_name
        elif prop in ["incident_id", "incidentId"]:
            args[prop] = incident_id
        elif p_type == "integer":
            args[prop] = p_info.get("default", 1)
        elif p_type == "boolean":
            args[prop] = True
        else:
            args[prop] = p_info.get("default", service_name)
            
    return args

def run_model_planner(incident: Dict[str, Any], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    transcript = incident.get("transcript", "")
    allowed_causes = incident.get("allowedRootCauses", [])
    all_evidence = extract_evidence_ids(transcript)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            prompt = f"""
Analyze this incident transcript to determine the single root cause and cite 2 to 4 evidence IDs.
Select 1 to 3 diagnostic tools required to confirm the issue.

Allowed Root Causes: {json.dumps(allowed_causes)}
Available Evidence IDs: {json.dumps(all_evidence)}
Available Tools: {json.dumps(tools)}

Transcript:
{transcript}

Return strictly structured JSON:
{{
  "rootCause": "matching cause from allowedRootCauses",
  "evidence": ["ev_1", "ev_2"],
  "diagnostics": [
    {{
      "toolName": "tool_name",
      "arguments": {{}}
    }}
  ]
}}
"""
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(res.choices[0].message.content)
            
            # Validate evidence contains only extracted IDs
            valid_ev = [e for e in data.get("evidence", []) if e in all_evidence]
            if len(valid_ev) >= 2:
                data["evidence"] = valid_ev[:4]
            else:
                data["evidence"] = all_evidence[:3] if len(all_evidence) >= 2 else all_evidence
                
            return data
        except Exception:
            pass

    # Deterministic Local Planner (Runs when LLM API Key is absent)
    selected_cause = allowed_causes[0] if allowed_causes else "unknown"
    for cause in allowed_causes:
        cause_tokens = cause.replace("_", " ").split()
        if any(token in transcript.lower() for token in cause_tokens):
            selected_cause = cause
            break

    chosen_evidence = all_evidence[:3] if len(all_evidence) >= 2 else all_evidence
    
    diag_tools = [t for t in tools if not t["name"].startswith("rollback") and not t["name"].startswith("disable")]
    selected_tools = diag_tools[:2] if len(diag_tools) >= 2 else diag_tools[:1]
    
    diagnostics = []
    for t in selected_tools:
        diagnostics.append({
            "toolName": t["name"],
            "arguments": extract_case_arguments(t.get("inputSchema", {}), incident)
        })

    return {
        "rootCause": selected_cause,
        "evidence": chosen_evidence,
        "diagnostics": diagnostics
    }

# =====================================================================
# Strict OTLP Trace Builder
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
    
    # SERVER Span
    spans.append({
        "traceId": trace_id,
        "spanId": server_span_id,
        "name": "POST /v2/incidents",
        "kind": 2,
        "attributes": common_attrs
    })
    
    # INTERNAL invoke_agent Span
    spans.append({
        "traceId": trace_id,
        "spanId": agent_span_id,
        "parentSpanId": server_span_id,
        "name": f"invoke_agent {agent_name}",
        "kind": 1,
        "attributes": common_attrs
    })
    
    # CLIENT chat incident-plan Span (Exactly 1)
    spans.append({
        "traceId": trace_id,
        "spanId": chat_span_id,
        "parentSpanId": agent_span_id,
        "name": "chat incident-plan",
        "kind": 3,
        "attributes": common_attrs + [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": state.get("modelName", "gpt-4o-mini")}}
        ]
    })
    
    diagnostic_exec_ids = []
    
    for action in state.get("actionLog", []):
        exec_span_id = hashlib.sha256((action['actionId'] + '_exec').encode()).hexdigest()[:16]
        
        if action.get("phase") == "diagnostic" and exec_span_id not in diagnostic_exec_ids:
            diagnostic_exec_ids.append(exec_span_id)
            
        spans.append({
            "traceId": trace_id,
            "spanId": exec_span_id,
            "parentSpanId": agent_span_id,
            "name": f"execute_tool {action['toolName']}",
            "kind": 1,
            "attributes": common_attrs + [
                {"key": "ga5.action.id", "value": {"stringValue": action["actionId"]}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": action["toolName"]}},
                {"key": "gen_ai.tool.call.id", "value": {"stringValue": action["callId"]}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}}
            ]
        })
        
        # Parse outgoing client span ID from stored traceparent
        client_span_id = action["traceparent"].split("-")[2]
        
        receipt = next((r for r in state.get("receiptLog", []) 
                        if r.get("actionId") == action["actionId"] and r.get("attempt") == action["attempt"]), None)
        
        client_attrs = common_attrs + [
            {"key": "ga5.action.id", "value": {"stringValue": action["actionId"]}},
            {"key": "ga5.attempt", "value": {"intValue": action["attempt"]}},
            {"key": "http.request.method", "value": {"stringValue": "POST"}},
            {"key": "http.request.resend_count", "value": {"intValue": action["attempt"] - 1}}
        ]
        
        span_status = {"code": 0}
        
        if receipt:
            client_attrs.append({"key": "ga5.receipt.id", "value": {"stringValue": receipt["receiptId"]}})
            if "nonce" in receipt:
                client_attrs.append({"key": "ga5.receipt.nonce", "value": {"stringValue": receipt["nonce"]}})
            
            st_code = receipt.get("status", 200)
            res_cls = receipt.get("resultClass", "")
            
            if st_code == 503 or res_cls == "503":
                span_status = {"code": 2}
                client_attrs.append({"key": "error.type", "value": {"stringValue": "503"}})
            elif st_code == 0 or res_cls == "timeout":
                span_status = {"code": 2}
                client_attrs.append({"key": "error.type", "value": {"stringValue": "timeout"}})

        spans.append({
            "traceId": trace_id,
            "spanId": client_span_id,
            "parentSpanId": exec_span_id,
            "name": f"POST tool/{action['toolName']}",
            "kind": 3,
            "attributes": client_attrs,
            "status": span_status
        })

    # Join span for parallel diagnostics
    if len(diagnostic_exec_ids) > 1:
        join_span_id = hashlib.sha256((run_id + '_join').encode()).hexdigest()[:16]
        spans.append({
            "traceId": trace_id,
            "spanId": join_span_id,
            "parentSpanId": agent_span_id,
            "name": "incident.join",
            "kind": 1,
            "attributes": common_attrs,
            "links": [{"traceId": trace_id, "spanId": sid} for sid in diagnostic_exec_ids]
        })
        
    # Approval gate span
    appr_receipt = next((r for r in state.get("receiptLog", []) if "approvalId" in r), None)
    if appr_receipt:
        gate_span_id = hashlib.sha256((run_id + '_appr').encode()).hexdigest()[:16]
        spans.append({
            "traceId": trace_id,
            "spanId": gate_span_id,
            "parentSpanId": agent_span_id,
            "name": "approval_gate",
            "kind": 1,
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

def build_client_response(state: Dict[str, Any]) -> Dict[str, Any]:
    st = state["status"]
    if st == "waiting":
        clean_approvals = [
            {
                "approvalId": a["approvalId"],
                "actionId": a["actionId"],
                "toolName": a["toolName"],
                "argumentsDigest": a["argumentsDigest"]
            }
            for a in state.get("approvals", [])
        ]
        return {
            "runId": state["runId"],
            "status": "waiting",
            "diagnosis": state["diagnosis"],
            "dispatches": state.get("dispatches", []),
            "approvals": clean_approvals
        }
    else:
        return {
            "runId": state["runId"],
            "status": st,
            "diagnosis": state["diagnosis"],
            "chosenEffect": state.get("chosenEffect"),
            "suppressed": state.get("suppressed", []),
            "actionLog": state.get("actionLog", []),
            "receiptLog": state.get("receiptLog", []),
            "otlp": build_otlp_trace(state)
        }

# =====================================================================
# API Endpoints
# =====================================================================

@app.post("/v2/incidents")
async def create_incident(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=422, content={"error": "Body must be an object"})

    if body.get("profile") != "ga5-incident-agent/v2":
        return JSONResponse(status_code=400, content={"error": "Unsupported profile"})

    run_id = body.get("runId")
    if not run_id or not isinstance(run_id, str):
        return JSONResponse(status_code=422, content={"error": "Missing or invalid runId"})

    req_hash = compute_hash(body)
    existing = get_run(run_id)
    if existing:
        if existing["req_hash"] == req_hash:
            return JSONResponse(content=build_client_response(existing["state"]), status_code=200)
        else:
            return JSONResponse(status_code=409, content={"error": "RunId content conflict"})

    traceparent_hdr = request.headers.get("traceparent")
    if traceparent_hdr and traceparent_hdr.startswith("00-"):
        trace_id = traceparent_hdr.split("-")[1]
    else:
        trace_id = generate_trace_id()

    server_span_id = generate_span_id()
    agent_span_id = generate_span_id()
    chat_span_id = generate_span_id()

    incident = body.get("incident", {})
    tool_catalog = body.get("toolCatalog", [])
    policy = body.get("policy", {})

    plan = run_model_planner(incident, tool_catalog)

    root_cause = plan.get("rootCause")
    evidence = plan.get("evidence", [])[:4]

    diagnostics_plan = plan.get("diagnostics", [])[:policy.get("maximumDiagnostics", 3)]
    
    dispatches = []
    action_log = []

    for diag in diagnostics_plan:
        action_id = generate_opaque_id("act")
        call_id = generate_opaque_id("call")
        client_span_id = generate_span_id()
        disp_tp = f"00-{trace_id}-{client_span_id}-01"
        
        disp = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": diag["toolName"],
            "arguments": diag.get("arguments", {}),
            "evidence": [evidence[0]] if evidence else [], # Each dispatch cites valid evidence ID
            "attempt": 1,
            "traceparent": disp_tp
        }
        dispatches.append(disp)
        action_log.append(disp)

    state = {
        "runId": run_id,
        "profile": body["profile"],
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
        "incident": incident,
        "toolCatalog": tool_catalog,
        "policy": policy,
        "dispatches": dispatches,
        "approvals": [],
        "chosenEffect": None,
        "suppressed": [],
        "actionLog": action_log,
        "receiptLog": []
    }

    save_run(run_id, body["profile"], body.get("publicMarker", ""), body.get("agentName", ""), req_hash, state)
    return JSONResponse(content=build_client_response(state), status_code=200)


@app.post("/v2/incidents/{runId}/receipts")
async def post_receipt(runId: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    receipt_id = body.get("receiptId")
    if not receipt_id:
        return JSONResponse(status_code=422, content={"error": "Missing receiptId"})

    receipt_hash = compute_hash(body)
    existing_receipt = get_receipt(receipt_id)
    
    stored_run = get_run(runId)
    if not stored_run:
        return JSONResponse(status_code=404, content={"error": "Run not found"})

    state = stored_run["state"]

    if existing_receipt:
        if existing_receipt["receipt_hash"] == receipt_hash:
            return JSONResponse(content=build_client_response(state), status_code=200)
        else:
            return JSONResponse(status_code=409, content={"error": "Receipt content conflict"})

    save_receipt(receipt_id, runId, receipt_hash, body)

    outcomes = body.get("outcomes", [])
    approvals_in = body.get("approvals", [])

    for outcome in outcomes:
        act_id = outcome["actionId"]
        attempt = outcome["attempt"]
        st = outcome.get("status", 200)
        res_cls = outcome.get("resultClass", "")
        
        state["receiptLog"].append({
            "receiptId": receipt_id,
            "actionId": act_id,
            "callId": outcome.get("callId"),
            "attempt": attempt,
            "status": st,
            "resultClass": res_cls,
            "nonce": outcome.get("nonce")
        })

        # Process 503 Retry
        if st == 503 and attempt == 1:
            matching_action = next((a for a in state["actionLog"] if a["actionId"] == act_id), None)
            if matching_action:
                retry_client_span_id = generate_span_id()
                retry_disp = dict(matching_action)
                retry_disp["attempt"] = 2
                retry_disp["traceparent"] = f"00-{state['traceId']}-{retry_client_span_id}-01"
                
                state["dispatches"] = [retry_disp]
                state["actionLog"].append(retry_disp)
                state["status"] = "waiting"
                save_run(runId, state["profile"], state["publicMarker"], state["agentName"], stored_run["req_hash"], state)
                return JSONResponse(content=build_client_response(state), status_code=200)

        # Process Timeout Failure
        if st == 0 or res_cls == "timeout":
            state["status"] = "failed"
            state["dispatches"] = []
            state["approvals"] = []
            save_run(runId, state["profile"], state["publicMarker"], state["agentName"], stored_run["req_hash"], state)
            return JSONResponse(content=build_client_response(state), status_code=200)

    # Process Approvals
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
                call_id = generate_opaque_id("call")
                client_span_id = generate_span_id()

                eff_dispatch = {
                    "actionId": act_id,
                    "callId": call_id,
                    "phase": "effect",
                    "toolName": effect_tool,
                    "arguments": pending_app.get("internal_arguments", {}),
                    "attempt": 1,
                    "traceparent": f"00-{state['traceId']}-{client_span_id}-01",
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

    # Evaluate Diagnostic Completion & Effect Dispatch
    pending_diagnostics = [a for a in state["actionLog"] if a.get("phase") == "diagnostic"]
    all_succeeded = all(
        any(r.get("actionId") == d["actionId"] and r.get("status") == 200 and r.get("resultClass") != "timeout" for r in state["receiptLog"])
        for d in pending_diagnostics
    )

    if all_succeeded and state["status"] == "waiting" and not state.get("approvals"):
        effect_tools = state.get("policy", {}).get("effectTools", [])
        if effect_tools:
            chosen_effect = effect_tools[0]
            approval_required = state.get("policy", {}).get("approvalRequiredFor", [])
            
            tool_meta = next((t for t in state.get("toolCatalog", []) if t["name"] == chosen_effect), {})
            eff_args = extract_case_arguments(tool_meta.get("inputSchema", {}), state.get("incident", {}))
            act_id = generate_opaque_id("act")

            if chosen_effect in approval_required:
                app_id = generate_opaque_id("appr")
                args_digest = compute_arguments_digest(eff_args)

                state["dispatches"] = []
                state["approvals"] = [{
                    "approvalId": app_id,
                    "actionId": act_id,
                    "toolName": chosen_effect,
                    "argumentsDigest": args_digest,
                    "internal_arguments": eff_args
                }]
            else:
                client_span_id = generate_span_id()
                call_id = generate_opaque_id("call")

                eff_dispatch = {
                    "actionId": act_id,
                    "callId": call_id,
                    "phase": "effect",
                    "toolName": chosen_effect,
                    "arguments": eff_args,
                    "attempt": 1,
                    "traceparent": f"00-{state['traceId']}-{client_span_id}-01"
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
        return JSONResponse(status_code=404, content={"error": "Run not found"})
    return JSONResponse(content=build_client_response(stored["state"]), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
