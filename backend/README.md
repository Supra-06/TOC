# Backend: Anomaly Detection in Process Workflows Using Automata and ML

This directory houses the complete backend service for monitoring online shopping order workflows and detecting anomalous process executions using a dual-engine architecture:
1. **Finite State Machine (FSM):** Deterministic verification of structural process rules and transition logic.
2. **Machine Learning (Isolation Forest):** Unsupervised statistical anomaly detection for unusual sequences and behavioral outliers.

The results from both engines are synthesized by a business logic service layer and exposed via **FastAPI** REST endpoints for the React frontend.

---

## 1. Directory Structure

```text
backend/
├── main.py                     # Application entry point (FastAPI app instance & lifecycle)
├── requirements.txt            # Python dependencies
├── README.md                   # Backend documentation and architecture guide
│
├── core/                       # Global configuration and environment settings
│   ├── __init__.py
│   └── config.py               # Central settings (DB path, model path, CORS, hyperparameters)
│
├── api/                        # HTTP interface layer (FastAPI)
│   ├── __init__.py
│   └── routes.py               # API route definitions and endpoint handlers
│
├── fsm/                        # Deterministic Automata Engine
│   ├── __init__.py
│   ├── states.py               # Order lifecycle states and event definitions
│   ├── transitions.py          # State transition matrix & valid path rules
│   └── validator.py            # Sequential transition checker & violation detector
│
├── ml/                         # Statistical Machine Learning Engine
│   ├── __init__.py
│   ├── dataset/                # Dataset generation & data management
│   │   ├── __init__.py
│   │   ├── generator.py        # Synthetic normal and anomalous workflow trace generator
│   │   └── data/               # Raw & processed CSV / JSON dataset storage (.gitkeep)
│   ├── feature_extraction/     # Feature engineering
│   │   ├── __init__.py
│   │   └── extractor.py        # Converts event sequences into numeric feature vectors
│   ├── training/               # Model training pipeline
│   │   ├── __init__.py
│   │   └── train.py            # Isolation Forest training and hyperparameter tuning script
│   ├── inference/              # Inference engine
│   │   ├── __init__.py
│   │   └── predictor.py        # Real-time scoring using serialized models
│   └── models/                 # Model artifacts (.joblib, scalers) (.gitkeep)
│
├── services/                   # Business Logic & Orchestration
│   ├── __init__.py
│   └── anomaly_service.py      # Combines FSM and ML analyses into unified verdict
│
├── database/                   # Data persistence layer (SQLite + SQLAlchemy)
│   ├── __init__.py
│   ├── connection.py           # Database engine, sessionmaker, and session dependency
│   └── models.py               # SQLAlchemy ORM models (orders, audit logs, anomaly records)
│
├── schemas/                    # Data validation & serialization (Pydantic v2)
│   ├── __init__.py
│   ├── order.py                # Order trace input and summary schemas
│   └── anomaly.py              # FSM, ML, and combined analysis response schemas
│
└── tests/                      # Automated test suite (Pytest)
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures and test database setup
    ├── test_fsm.py             # Unit tests for FSM transitions and edge cases
    ├── test_ml.py              # Tests for dataset generation, feature extraction, and ML
    └── test_api.py             # Integration tests for FastAPI endpoints
```

---

## 2. Directory & Component Responsibilities

| Directory / File | Purpose |
| :--- | :--- |
| `main.py` | FastAPI application factory, CORS middleware configuration for React frontend, route inclusion, and startup/shutdown lifecycle hooks. |
| `core/config.py` | Centralized configuration managing file paths, database URLs, ML parameters (e.g., contamination factor), and server settings. |
| `api/` | RESTful API endpoints handling incoming HTTP requests, delegating to the service layer, and returning validated JSON responses. |
| `fsm/` | Implements the deterministic Finite State Machine. Contains state enumerations, the transition table, and the verification engine that flags illegal transitions (e.g., `Delivered` before `Paid`). |
| `ml/dataset/` | Generates realistic synthetic event logs of online shopping workflows (normal shopping journeys and injected anomalies) for training and benchmarking. |
| `ml/feature_extraction/` | Transforms variable-length categorical event sequences into fixed-dimension numerical vectors (e.g., transition frequencies, sequence length, n-grams, duration). |
| `ml/training/` | Trains the unsupervised `IsolationForest` model on workflow features and serializes the trained artifacts using `Joblib`. |
| `ml/inference/` | Loads saved model artifacts and computes anomaly scores for incoming workflow traces in real time. |
| `ml/models/` | Storage for `.joblib` model artifacts and vectorizer parameters (excluded from git tracking). |
| `services/anomaly_service.py` | The central brain orchestrating the dual-detection strategy. Runs an order trace through both FSM and ML, correlates the findings, and produces a definitive diagnosis. |
| `database/` | Manages local SQLite storage via SQLAlchemy ORM to persist order histories, submitted traces, and detection logs for auditability. |
| `schemas/` | Pydantic models enforcing strict type safety and auto-generating OpenAPI documentation for request bodies and response payloads. |
| `tests/` | Pytest test suite ensuring high reliability for FSM rule enforcement, feature vectorization math, ML scoring, and API contracts. |

---

## 3. Technology Stack

- **Python 3.10+**: Core programming language.
- **FastAPI**: Modern, high-performance web framework for building REST APIs with automatic OpenAPI (Swagger) documentation.
- **Uvicorn**: ASGI web server implementation.
- **Scikit-learn**: Unsupervised Machine Learning (`IsolationForest`) for anomaly detection.
- **Pandas & NumPy**: Data manipulation, synthetic trace generation, and matrix transformations.
- **Joblib**: Efficient serialization and deserialization of Python ML models and transformers.
- **SQLite**: Zero-configuration, file-based relational database included in Python.
- **SQLAlchemy 2.0**: Python SQL toolkit and Object Relational Mapper (ORM).
- **Pydantic v2**: Data parsing and validation for request/response schemas.
- **Pytest & HTTPX**: Testing framework and async HTTP test client.

---

## 4. Planned Workflow Lifecycle (E-Commerce Example)

A typical order workflow lifecycle progresses through defined stages:
```text
[Order Placed] ──> [Payment Pending] ──> [Payment Confirmed] ──> [Item Packed] ──> [Shipped] ──> [Delivered]
       │                    │                     │                   │
       └──> [Cancelled]     └──> [Payment Failed] └──> [Refunded]     └──> [Returned]
```

- **FSM Target:** Strictly checks legal transitions (e.g., cannot go from `Order Placed` directly to `Shipped`).
- **ML Target:** Detects unusual patterns that might technically be legal according to the FSM but represent rare, suspicious, or statistically anomalous behavioral clusters (e.g., excessive retries, skipping optional cart reviews, abnormal sequence duration).
- **Combined Service:** Flags whether a sequence is strictly invalid (FSM), statistically irregular (ML), or both.

