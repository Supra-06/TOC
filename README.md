# Anomaly Detection in Process Workflows Using Automata and ML

A college project monitoring online shopping order workflows using two complementary detection paradigms:

1. **Finite State Machine (FSM):** Deterministic verification of sequence validity and allowed lifecycle transitions.
2. **Machine Learning (Isolation Forest):** Unsupervised statistical anomaly detection on sequence features and behavioral patterns.

A unified **FastAPI** backend exposes both detection engines, with results stored in **SQLite** and ready for consumption by a future **React** frontend.

---

## Project Structure Overview

```text
TOC_PROJECT/
├── backend/                    # FastAPI backend service
│   ├── main.py                 # Application entry point
│   ├── requirements.txt        # Backend dependencies
│   ├── README.md               # Detailed backend documentation
│   ├── core/                   # Application configuration & settings
│   ├── api/                    # REST API routes
│   ├── fsm/                    # Finite State Machine implementation
│   ├── ml/                     # Dataset generation, feature extraction, training & inference
│   │   ├── dataset/            # Synthetic data generation and trace files
│   │   ├── feature_extraction/ # Event sequence vectorization
│   │   ├── training/           # Isolation Forest training pipeline
│   │   ├── inference/          # Anomaly scoring engine
│   │   └── models/             # Serialized model artifacts (.joblib)
│   ├── services/               # Combined analysis business logic (FSM + ML)
│   ├── database/               # SQLite connection & SQLAlchemy ORM models
│   ├── schemas/                # Pydantic request and response schemas
│   └── tests/                  # Backend test suite
└── .gitignore                  # Git ignore rules
```

For detailed backend architecture, installation instructions, API endpoints, and usage, see [backend/README.md](backend/README.md).

## Project Status

- Backend implementation: Complete
- FSM validation: Complete
- Isolation Forest pipeline: Complete
- Hybrid anomaly detection: Complete
- Database integration: Complete
- REST API: Complete
- End-to-end backend verification: Complete
- Frontend: Pending
