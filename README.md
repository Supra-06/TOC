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
│   └── tests/                  # Pytest test suite
└── .gitignore                  # Git ignore rules for Python, virtual environments, and caches
```

For detailed backend architecture and directory descriptions, see [backend/README.md](backend/README.md).

