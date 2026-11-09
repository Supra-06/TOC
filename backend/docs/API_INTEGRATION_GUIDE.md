# TOC Workflow Anomaly Detection — Frontend Integration Guide

This guide provides the authoritative specification for connecting frontend client applications (e.g., React, Next.js, Vite, or Vue dashboards) to the FastAPI backend service for **Anomaly Detection in Process Workflows Using Automata and Machine Learning**.

---

## 1. Backend Setup & Startup (Windows)

### Prerequisites
- Python 3.10+ (tested with Python 3.13)
- Windows PowerShell or Command Prompt
- Virtual environment at `.venv`

### Startup Commands
Activate the virtual environment and start the development server with Uvicorn:

```powershell
# From the project root (c:\Users\SUPRATEEK\TOC_PROJECT):
.\.venv\Scripts\Activate.ps1

# Run the Uvicorn ASGI server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The interactive OpenAPI documentation will be accessible at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON Spec: `http://127.0.0.1:8000/openapi.json`

---

## 2. Environment Variables and Configuration

The backend reads settings from environment variables with safe defaults:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `TOC_DATABASE_URL` | String | `sqlite:///backend/toc_workflow.db` | SQLAlchemy connection URL for SQLite database. |
| `TOC_DATABASE_ECHO` | Boolean | `False` | Log raw SQL statements to console (`true`/`false`). |
| `INIT_DB_ON_STARTUP` | Boolean | `False` | Automatically create database tables during ASGI lifespan startup. |
| `CORS_ORIGINS` | Comma-delimited | `http://localhost:3000,http://localhost:5173` | Allowed frontend web origins. |

---

## 3. Base URL, Versioning & CORS Policy

- **Base URL:** `http://127.0.0.1:8000`
- **API Version Prefix:** `/api/v1`
- **Operational Root Health:** `GET /health` (unversioned liveness probe)

### CORS Policy
- **Allowed Origins:** `http://localhost:3000` (React default) and `http://localhost:5173` (Vite default).
- **Credentials:** Allowed (`Access-Control-Allow-Credentials: true`). Wildcards (`*`) are disallowed.
- **Allowed Methods:** Explicitly restricted to `GET` and `POST`.
- **Allowed Headers:** Explicitly restricted to `Authorization` and `Content-Type`.

---

## 4. Verdicts, Workflow Automata, and ML Taxonomy

The engine evaluates event sequences using a hybrid two-stage decision tree combining a deterministic **Finite State Machine (FSM)** and an **Isolation Forest ML Model**.

### Approved 4-State Verdict Taxonomy

| Verdict | Meaning | `is_anomaly` | ML Evaluated? | Diagnostic Severity |
| :--- | :--- | :--- | :--- | :--- |
| `NORMAL` | Sequence is valid, complete (reaches a terminal state), and exhibits normal behavioral volume. | `False` | **Yes** | `INFO` |
| `RULE_VIOLATION` | Sequence violates formal FSM transition grammar or terminal state semantics. | `True` | **No** (Skipped) | `CRITICAL` |
| `STATISTICAL_ANOMALY` | Sequence satisfies FSM rules and reaches a terminal state, but exhibits statistical friction (e.g., payment retry loops). | `True` | **Yes** | `WARNING` |
| `VALID_INCOMPLETE` | Sequence is structurally valid according to FSM transitions, but represents an active, non-terminal workflow prefix. | `False` | **No** (Skipped) | `INFO` |

### Allowed FSM Alphabet ($\Sigma$)
1. `CONFIRM_PAYMENT`
2. `FAIL_PAYMENT`
3. `RETRY_PAYMENT`
4. `CANCEL_ORDER`
5. `PACK_ITEMS`
6. `SHIP_ITEMS`
7. `DELIVER_ITEMS`
8. `REFUND_ORDER`

### Terminal States ($F$)
- `DELIVERED` (completed successfully)
- `CANCELLED` (terminated early)
- `REFUNDED` (returned/refunded post-delivery)

---

## 5. REST API Endpoints Specification

### 5.1 Liveness Probes
- `GET /health`
- `GET /api/v1/health/live`
- **Response Code:** `200 OK`
- **Description:** Verifies HTTP container responsiveness without querying SQLite or loading the ML model into memory.

**Response Schema (`LivenessResponse`):**
```json
{
  "status": "alive"
}
```

---

### 5.2 Readiness Probes
- `GET /api/v1/health/ready`
- `GET /api/v1/health`
- **Response Codes:** `200 OK` (Ready), `503 Service Unavailable` (Not Ready)
- **Description:** Verifies database connectivity (`SELECT 1`) and model file artifact presence on disk without forced model loading.

**Response Schema (`ReadinessResponse`):**
```json
{
  "service_status": "OPERATIONAL",
  "dataset_spec_version": "1.4",
  "model_artifact_path": "C:\\Users\\SUPRATEEK\\TOC_PROJECT\\backend\\ml\\models\\isolation_forest.joblib",
  "model_artifact_exists": true,
  "supported_verdicts": [
    "NORMAL",
    "RULE_VIOLATION",
    "STATISTICAL_ANOMALY",
    "VALID_INCOMPLETE"
  ]
}
```

---

### 5.3 Single Order Trace Analysis
- `POST /api/v1/analyses`
- **Response Codes:** `201 Created`, `422 Unprocessable Content`, `500 Internal Server Error`
- **Description:** Submits a single order trace for hybrid evaluation and persists the audit record.

#### Request Formats (`WorkflowAnalysisRequest`):
Event sequences may be supplied as a semicolon-delimited string or as a list of event token strings:

```json
{
  "order_id": "ORD-2026-9041",
  "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
}
```
*Or with an array of strings:*
```json
{
  "order_id": "ORD-2026-9041",
  "event_sequence": ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
}
```

#### Successful Response (`AnalysisRecordResponse`):
```json
{
  "id": 1,
  "order_id": "ORD-2026-9041",
  "input_format": "DELIMITED_STRING",
  "raw_input_payload": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
  "verdict": "NORMAL",
  "is_anomaly": false,
  "anomaly_type": null,
  "summary": "Workflow completed normally and terminated at 'DELIVERED' with a typical behavioral profile.",
  "fsm_evaluation": {
    "is_rule_valid": true,
    "is_complete": true,
    "final_state": "DELIVERED",
    "total_steps": 4,
    "successful_transitions": 4,
    "error": null
  },
  "ml_evaluation": {
    "prediction": "NORMAL",
    "is_ml_anomaly": false,
    "anomaly_score": -0.2154,
    "raw_decision_score": 0.1842,
    "features": {
      "seq_length": 4.0,
      "count_CONFIRM_PAYMENT": 1.0,
      "count_FAIL_PAYMENT": 0.0,
      "count_RETRY_PAYMENT": 0.0,
      "count_CANCEL_ORDER": 0.0,
      "count_PACK_ITEMS": 1.0,
      "count_SHIP_ITEMS": 1.0,
      "count_DELIVER_ITEMS": 1.0,
      "count_REFUND_ORDER": 0.0,
      "retry_ratio": 0.0,
      "payment_failure_rate": 0.0,
      "bigram_FAIL_to_RETRY": 0.0,
      "bigram_PACK_to_SHIP": 1.0,
      "unique_event_count": 4.0
    }
  },
  "ml_skip_reason": null,
  "diagnostics": {
    "severity": "INFO",
    "root_cause": "Order completed normally terminating in terminal state 'DELIVERED'.",
    "key_factors": [
      "FSM validated without errors",
      "Terminated in valid state DELIVERED",
      "Isolation Forest score within normal baseline (-0.2154)"
    ],
    "recommended_action": "Standard order execution following expected baseline paths. No operational intervention required."
  },
  "created_at": "2026-11-09T14:15:30.123456Z"
}
```

---

### 5.4 Batch Workflow Trace Analysis
- `POST /api/v1/analyses/batch`
- **Response Codes:** `201 Created`, `422 Unprocessable Content`, `500 Internal Server Error`
- **Batch Size Limit:** 1 to 500 traces per request.
- **Atomicity:** All traces are evaluated and persisted in a single transaction. If any failure occurs, the transaction rolls back completely with zero partial writes.

#### Request Formats (`BatchAnalysisRequest`):
Supports mixed representations:
1. Structured object (`{"order_id": "...", "event_sequence": "..."}`)
2. Plain semicolon-delimited string
3. List of event strings

```json
{
  "traces": [
    {
      "order_id": "ORD-B01",
      "event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS"
    },
    "CONFIRM_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
    ["CONFIRM_PAYMENT", "PACK_ITEMS"]
  ]
}
```

#### Successful Response (`BatchAnalysisResponse`):
```json
{
  "results": [
    {
      "id": 1,
      "order_id": "ORD-B01",
      "input_format": "DELIMITED_STRING",
      "raw_input_payload": "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
      "verdict": "NORMAL",
      "is_anomaly": false,
      "anomaly_type": null,
      "summary": "Workflow completed normally...",
      "fsm_evaluation": { ... },
      "ml_evaluation": { ... },
      "ml_skip_reason": null,
      "diagnostics": { ... },
      "created_at": "2026-11-09T14:20:00.000000Z"
    }
  ],
  "summary": {
    "total_orders": 3,
    "normal_count": 1,
    "rule_violation_count": 0,
    "statistical_anomaly_count": 1,
    "valid_incomplete_count": 1,
    "total_anomalies": 1,
    "anomaly_rate": 0.3333
  }
}
```

---

### 5.5 Single Record Retrieval
- `GET /api/v1/analyses/{analysis_id}`
- **Parameters:** `analysis_id` (integer path parameter, `gt=0`)
- **Response Codes:** `200 OK`, `404 Not Found`, `500 Internal Server Error`
- **Response:** `AnalysisRecordResponse`

---

### 5.6 Order History Timeline
- `GET /api/v1/orders/{order_id}/analyses`
- **Parameters:**
  - `order_id` (string path parameter, 1-64 characters)
  - `limit` (query parameter, default: 50, max: 1000)
- **Response Codes:** `200 OK`, `500 Internal Server Error`
- **Ordering:** Chronological descending (`created_at DESC, id DESC`).

#### Response Schema (`OrderHistoryResponse`):
```json
{
  "order_id": "ORD-2026-9041",
  "returned_count": 2,
  "analyses": [
    {
      "id": 2,
      "order_id": "ORD-2026-9041",
      "verdict": "NORMAL",
      ...
    },
    {
      "id": 1,
      "order_id": "ORD-2026-9041",
      "verdict": "VALID_INCOMPLETE",
      ...
    }
  ]
}
```

---

### 5.7 Filtered and Paginated Analyses
- `GET /api/v1/analyses`
- **Query Parameters:**
  - `skip` (integer, default: 0, minimum: 0)
  - `limit` (integer, default: 50, minimum: 1, maximum: 1000)
  - `verdict` (optional string, one of `NORMAL`, `RULE_VIOLATION`, `STATISTICAL_ANOMALY`, `VALID_INCOMPLETE`)
  - `is_anomaly` (optional boolean, `true` or `false`)

#### Response Schema (`PaginatedAnalysesResponse`):
```json
{
  "items": [ ... ],
  "total": 142,
  "skip": 0,
  "limit": 50
}
```

---

### 5.8 Database Summary Metrics
- `GET /api/v1/metrics`
- **Response Code:** `200 OK`
- **Description:** Aggregates database-wide trace counts and overall anomaly proportion.

#### Response Schema (`DatabaseSummaryMetricsResponse`):
```json
{
  "total_analyses": 1250,
  "normal_count": 875,
  "rule_violation_count": 250,
  "statistical_anomaly_count": 100,
  "valid_incomplete_count": 25,
  "total_anomalies": 350,
  "anomaly_rate": 0.28,
  "unique_orders_count": 1180
}
```

---

## 6. Standardized Error Envelope

All HTTP client and server errors (`400`, `404`, `422`, `500`, `503`) follow the unified nested error envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "details": [
      {
        "type": "string_type",
        "loc": ["body", "event_sequence"],
        "msg": "Input should be a valid string",
        "input": 12345
      }
    ]
  }
}
```

### Standard Error Codes
- `VALIDATION_ERROR` (HTTP 422): Schema or parameter validation failed.
- `NOT_FOUND` (HTTP 404): Resource ID does not exist.
- `SERVICE_UNAVAILABLE` (HTTP 503): Database or model dependencies are unavailable.
- `MAPPING_ERROR` (HTTP 500): Corrupted database record could not be mapped to schema.
- `INTERNAL_SERVER_ERROR` (HTTP 500): Safe fallback for unhandled server exceptions without leaking internal stack traces.

---

## 7. Frontend Integration Code Examples

### 7.1 Single Workflow Trace Analysis (`fetch`)

```typescript
interface SingleAnalysisRequest {
  order_id?: string;
  event_sequence: string | string[];
}

async function analyzeWorkflow(payload: SingleAnalysisRequest) {
  const response = await fetch("http://127.0.0.1:8000/api/v1/analyses", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // "Authorization": "Bearer <token>"  // Optional
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`[${errorData.error.code}] ${errorData.error.message}`);
  }

  return await response.json();
}

// Example invocation
analyzeWorkflow({
  order_id: "ORD-9912",
  event_sequence: "CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS",
}).then((data) => {
  console.log("Verdict:", data.verdict);
  console.log("Anomaly Score:", data.ml_evaluation?.anomaly_score);
});
```

---

### 7.2 Batch Analysis Submission (`fetch`)

```typescript
async function analyzeBatch(traces: Array<string | string[] | { order_id?: string; event_sequence: string | string[] }>) {
  const response = await fetch("http://127.0.0.1:8000/api/v1/analyses/batch", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ traces }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`Batch failed: ${errorData.error.message}`);
  }

  return await response.json();
}
```

---

### 7.3 Fetching Summary Metrics for Dashboard KPI Cards

```typescript
async function fetchDashboardMetrics() {
  const response = await fetch("http://127.0.0.1:8000/api/v1/metrics");
  if (!response.ok) throw new Error("Could not retrieve metrics");
  const metrics = await response.json();
  
  return {
    totalTraces: metrics.total_analyses,
    anomalyRate: (metrics.anomaly_rate * 100).toFixed(1) + "%",
    ruleViolations: metrics.rule_violation_count,
    statisticalAnomalies: metrics.statistical_anomaly_count,
  };
}
```

---

## 8. Frontend UI Rendering Recommendations

### 8.1 Status Badges & Colors

| Verdict | Badge Text | Recommended Color |
| :--- | :--- | :--- |
| `NORMAL` | Normal Execution | Green (`#10B981` / Emerald) |
| `RULE_VIOLATION` | Rule Violation | Red (`#EF4444` / Crimson) |
| `STATISTICAL_ANOMALY` | Statistical Anomaly | Amber/Orange (`#F59E0B` / Amber) |
| `VALID_INCOMPLETE` | In-Progress | Blue/Cyan (`#3B82F6` / Sky) |

### 8.2 Rendering FSM Errors
When `fsm_evaluation.error` is present:
- Display the `error_type` (e.g., `INVALID_TRANSITION`, `PREMATURE_TERMINATION`).
- Highlight the `failed_event` and `failed_step` in a visual trace progress bar.
- Show the `state_at_failure` and suggest valid actions from `allowed_events`.

### 8.3 Rendering ML Anomaly Scores
When `ml_evaluation` is present:
- **Decision Score Bar:** Render `anomaly_score` on a normalized gauge.
- **Top Contributing Features:** Inspect the 14 features in `ml_evaluation.features` (e.g., high `retry_ratio` or `payment_failure_rate`) to explain the anomaly reason.

### 8.4 Rendering Operational Diagnostics
- Render the `diagnostics.severity` badge (`CRITICAL`, `WARNING`, `INFO`).
- Display `diagnostics.root_cause` in the primary summary box.
- Render `diagnostics.key_factors` as an enumerated bulleted list.
- Display `diagnostics.recommended_action` in a callout banner for triage operators.

