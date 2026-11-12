# Anomaly Detection in Process Workflows Using Automata and ML

A college Theory of Computation project monitoring online shopping order workflows using two complementary detection approaches:

- **Finite State Machine (FSM):** Deterministic verification of workflow validity and permitted state transitions based on formal automata theory.
- **Machine Learning (Isolation Forest):** Unsupervised detection of statistically unusual patterns in valid and complete workflow sequences.

A unified **FastAPI** backend coordinates both detection engines, persists analysis records and operational metrics in **SQLite**, and exposes REST APIs for a future **React** frontend.

---

## Key Features

The completed backend provides the following core capabilities:

- **FSM-Based Workflow Validation:** Deterministic verification of order event sequences against a formal 8-state, 8-event, 11-transition automaton.
- **Synthetic Dataset Generation:** Generation and partitioning of 5,000 labeled workflow traces across normal, statistical anomaly, and rule violation categories.
- **Feature Extraction:** Extraction of exactly 14 numerical features from event sequences (lengths, event counts, transition bigrams, retry ratios, and failure rates).
- **Isolation Forest ML Engine:** Unsupervised anomaly detection trained exclusively on normal baseline workflows with real-time inference.
- **Hybrid FSM + ML Detection:** Sequential two-stage evaluation logic ensuring structural validity preempts statistical scoring.
- **Four-Verdict Taxonomy:** Categorizes workflow traces into `NORMAL`, `RULE_VIOLATION`, `STATISTICAL_ANOMALY`, or `VALID_INCOMPLETE`.
- **Evidence-Based Diagnostics:** Generates root-cause explanations, contributing factors, and operational recommendations (`CRITICAL`, `WARNING`, `INFO`).
- **SQLite Persistence:** SQLAlchemy 2.0 ORM storage supporting atomic single and batch transactions, order history tracking, and database-wide metrics aggregation.
- **Production REST API:** Endpoints for single-trace analysis, batch analysis (up to 500 traces), historical timeline retrieval, pagination, filtering, metrics, and health/readiness checks.

---

## Project Structure Overview

```text
TOC_PROJECT/
├── README.md                   # Root project introduction and repository overview
├── .gitignore                  # Git ignore rules
└── backend/                    # FastAPI backend service
    ├── README.md               # Comprehensive backend architecture and execution guide
    ├── main.py                 # FastAPI application factory, lifespan, and CORS middleware
    ├── requirements.txt        # Backend Python dependencies
    ├── core/                   # Application configuration, paths, and environment settings
    ├── api/                    # REST API routes, dependencies, and ORM-to-Pydantic mapping
    ├── fsm/                    # Finite State Machine implementation and validator
    ├── ml/                     # Machine learning pipeline
    │   ├── dataset/            # Synthetic dataset generator and CSV storage
    │   ├── feature_extraction/ # 14-feature sequence vectorizer
    │   ├── training/           # Isolation Forest training script
    │   ├── inference/          # Online scoring and prediction engine
    │   ├── evaluation/         # Model evaluation and benchmark metrics
    │   └── models/             # Serialized model artifacts (.joblib, metadata.json)
    ├── pipeline/               # Hybrid analyzer coordinating FSM rules and ML inference
    ├── services/               # Domain service layer, diagnostic enrichment, and batch processing
    ├── database/               # SQLite connection, SQLAlchemy ORM models, and repository operations
    ├── schemas/                # Pydantic v2 request, response, and error schemas
    ├── tests/                  # Automated test suite (289 tests across 15 modules)
    └── docs/                   # Formal specifications and API integration guide
```

---

## Technology Stack

The project backend uses the following primary technologies:

- **Python (3.10+)**: Core programming language.
- **FastAPI**: Modern, high-performance web framework for RESTful APIs.
- **Uvicorn**: ASGI web server implementation.
- **Scikit-learn**: Isolation Forest unsupervised anomaly detection model.
- **Pandas & NumPy**: Data processing, matrix transformations, and dataset generation.
- **Joblib**: Model artifact serialization and loading.
- **SQLite**: Local relational database engine.
- **SQLAlchemy (2.0)**: Object Relational Mapper and database transaction management.
- **Pydantic (v2)**: Strict data parsing, request validation, and response serialization.
- **Pytest & Unittest**: Comprehensive automated test frameworks.
- **HTTPX**: HTTP client library supporting API integration testing.

---

## Project Status

| Component | Status | Description |
| :--- | :---: | :--- |
| **Backend Implementation** | Complete | Full FastAPI application with lifecycle management and CORS |
| **FSM Validation** | Complete | Deterministic 8-state automaton verifying workflow transitions |
| **Synthetic Dataset Generation** | Complete | 5,000-trace dataset generation and allocation |
| **ML Pipeline** | Complete | 14-feature extraction, Isolation Forest training, and online scoring |
| **Hybrid Anomaly Detection** | Complete | Two-stage decision engine synthesizing FSM and ML verdicts |
| **Database Integration** | Complete | SQLite persistence, atomic transactions, history, and metrics queries |
| **REST API Development** | Complete | 10 endpoints for operational health, analysis, retrieval, and metrics |
| **End-to-End Verification** | Complete | Verified test suites and isolated runtime smoke tests |
| **React Frontend** | Pending | Dashboard UI and visual workflow explorer (future phase) |

---

## Documentation

For technical specifications, local execution instructions, and integration details:

- **[Backend Documentation](backend/README.md):** Detailed guide to backend architecture, mathematical formulation, local setup instructions for Windows PowerShell, Uvicorn server execution, and test commands.
- **[API Integration Guide](backend/docs/API_INTEGRATION_GUIDE.md):** Complete frontend integration reference covering API endpoints, request and response JSON schemas, error envelopes, CORS policies, environment variables, and TypeScript `fetch()` code examples.

---

## Testing Summary

The backend codebase has been verified with **289 automated tests** passing with zero failures or regressions:

```powershell
python -m unittest discover -s backend/tests -v
# Ran 289 tests in 3.382s. OK
```

The test suite covers:
- **FSM Validation:** State transitions, invalid step detection, and allowed event evaluation.
- **Dataset Generation:** Data consistency, class allocations, and CSV schema verification.
- **Feature Extraction:** 14-feature numerical correctness and non-finite value handling.
- **ML Training, Inference & Evaluation:** Model fitting, metadata validation, decision scores, and benchmarks.
- **Hybrid Analysis:** Two-stage conditional routing across all four verdicts.
- **Service & Database Layers:** Diagnostic enrichment, atomic transactions, rollback, and repository queries.
- **API Schemas & Routes:** Pydantic validation, CORS middleware, and HTTP response envelopes.
- **End-to-End Integration:** Full API-to-database integration and real-model smoke tests.

*(Note: These results represent backend automated testing; frontend UI testing is pending future frontend development.)*
