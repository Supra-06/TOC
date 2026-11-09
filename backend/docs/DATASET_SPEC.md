# Workflow Trace Dataset Specification

**Project:** Anomaly Detection in Process Workflows Using Automata and ML  
**Domain:** Online Shopping Order Management Workflow  
**Document Version:** 1.4  
**Status:** Approved Specification (Revised)  

---

## 1. Purpose & Architectural Context

This document specifies the synthetic dataset design for evaluating our dual-engine anomaly detection system:

1. **Finite State Machine (FSM):** Deterministically verifies sequence syntax, legal transitions, and workflow termination rules.
2. **Machine Learning (Isolation Forest):** Unsupervised statistical anomaly detection identifying rare sequences, behavioral outliers, and frequency distortions that are structurally legal.

The dataset must support both mechanisms without confusing their independent responsibilities. Rule violations are caught deterministically by the FSM, while the Isolation Forest detects behavioral and frequency anomalies.

```text
                                  ┌─────────────────────────────────────────┐
                                  │   Synthetic Dataset Pool (5,000 Traces) │
                                  └────────────────────┬────────────────────┘
                                                       │
                       ┌───────────────────────────────┴───────────────────────────────┐
                       ▼                                                               ▼
        ┌─────────────────────────────┐                                 ┌─────────────────────────────┐
        │   Training Set (2,800)      │                                 │      Test Pool (2,200)      │
        │   100% Clean NORMAL Traces  │                                 │                             │
        └──────────────┬──────────────┘                                 └──────────────┬──────────────┘
                       │                                                               │
                       ▼                                               ┌───────────────┴───────────────┐
        ┌─────────────────────────────┐                                ▼                               ▼
        │    Fit Isolation Forest     │                 ┌─────────────────────────────┐ ┌─────────────────────────────┐
        │    (contamination="auto")   │                 │    Primary ML Test Set      │ │    FSM Benchmark Set        │
        └─────────────────────────────┘                 │   1,200 FSM-Valid Traces    │ │   1,000 RULE_VIOLATION      │
                                                        │   • 700 NORMAL (False)      │ │   Traces (is_rule_valid=    │
                                                        │   • 500 STATISTICAL_        │ │   False)                    │
                                                        │     ANOMALY (True)          │ └─────────────────────────────┘
                                                        └─────────────────────────────┘
```

---

## 2. Dataset Categories & Taxonomy

Every trace in the dataset belongs to exactly one of three distinct categories:

```text
                            ┌───────────────────────────────────┐
                            │    Synthetic Event Sequences      │
                            └─────────────────┬─────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
          ┌───────────────────────┐                       ┌───────────────────────┐
          │   FSM Rule Valid      │                       │   FSM Rule Invalid    │
          │  (is_rule_valid=True) │                       │ (is_rule_valid=False) │
          └───────────┬───────────┘                       └───────────┬───────────┘
                      │                                               │
         ┌────────────┴────────────┐                                  │
         ▼                         ▼                                  ▼
┌──────────────────┐      ┌──────────────────┐               ┌──────────────────┐
│ Category 1:      │      │ Category 3:      │               │ Category 2:      │
│ NORMAL           │      │ STATISTICAL_     │               │ RULE_VIOLATION   │
│                  │      │ ANOMALY          │               │                  │
│ • Standard path  │      │ • Valid in FSM   │               │ • Out-of-order   │
│ • Typical volume │      │ • Excessive loops│               │ • Skipped steps  │
│ • ML: False      │      │ • 3-6 retry loops│               │ • Post-terminal  │
│                  │      │ • ML: True       │               │ • ML: null (N/A) │
└──────────────────┘      └──────────────────┘               └──────────────────┘
```

### 2.1 Category 1: Normal Workflow Sequences (`NORMAL`)
- **Definition:** Standard consumer workflows that progress through the valid transitions defined in `WORKFLOW_SPEC.md` without unusual repetitions or abnormal path variations. Every generated normal sequence runs to completion, terminating in one of the defined terminal states (`DELIVERED`, `CANCELLED`, `REFUNDED`).
- **Characteristics:**
  - `is_rule_valid = True`
  - `is_complete = True`
  - `is_ml_anomaly = False`
- **Representative Examples:**
  The following represent common completed paths through the workflow:
  - **Standard Full Fulfillment:**
    `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]`
  - **Immediate Pre-Payment Cancellation:**
    `[CANCEL_ORDER]`
  - **Routine Single Payment Retry Followed by Fulfillment:**
    `[FAIL_PAYMENT, RETRY_PAYMENT, CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]`
  - **Paid Order Cancellation Before Packaging (Refunded):**
    `[CONFIRM_PAYMENT, REFUND_ORDER]`
  - **Packed Order Cancellation at Warehouse (Refunded):**
    `[CONFIRM_PAYMENT, PACK_ITEMS, REFUND_ORDER]`

*(Note: These examples illustrate typical operational trajectories. Actual generated frequencies emerge from the stochastic random-walk branch probabilities defined in Section 4.1, rather than predetermined fixed percentages).*

### 2.2 Category 2: Rule-Invalid Sequences (`RULE_VIOLATION`)
- **Definition:** Sequences that violate the formal transition function $\delta: Q \times \Sigma \rightharpoonup Q$. These represent impossible operational transitions, security tampering, system bugs, or protocol skips.
- **Characteristics:**
  - `is_rule_valid = False`
  - `is_complete = False`
  - `is_ml_anomaly = null` (Not applicable; evaluated strictly by the FSM, not positive examples in primary ML metrics)
- **Representative Patterns:**
  - **Prerequisite Step Skipping (Direct Dispatch):**
    `[CONFIRM_PAYMENT, SHIP_ITEMS, DELIVER_ITEMS]` *(Packing was skipped)*
  - **Unearned Refund (Refunding an Unpaid Order):**
    `[REFUND_ORDER]` *(Cannot refund an order in `ORDER_CREATED`)*
  - **Premature Delivery Attempt:**
    `[DELIVER_ITEMS, CONFIRM_PAYMENT]` *(Delivering before payment/packing)*
  - **Post-Terminal Execution (Zombie Mutation):**
    `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS, CANCEL_ORDER]` *(Action after `DELIVERED`)*
  - **Duplicate Action Violation (Double Charge):**
    `[CONFIRM_PAYMENT, CONFIRM_PAYMENT, PACK_ITEMS]` *(Duplicate payment in `PAID`)*
  - **Cancellation During Courier Transit:**
    `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, CANCEL_ORDER]` *(Cannot cancel in `SHIPPED`)*
  - **Unknown Event Injection:**
    `[CONFIRM_PAYMENT, BYPASS_AUTH_TOKEN, PACK_ITEMS]` *(Event not in alphabet $\Sigma$)*

### 2.3 Category 3: Statistically Unusual but Valid Sequences (`STATISTICAL_ANOMALY`)
- **Definition:** Sequences that are **100% structurally valid** according to the FSM transition table, but exhibit extreme, suspicious, or abnormal behavioral loops that deviate significantly from standard user behavior.
- **Characteristics:**
  - `is_rule_valid = True`
  - `is_complete = True`
  - `is_ml_anomaly = True`
- **Why this category is critical:**
  > [!IMPORTANT]
  > The core academic thesis of this project is that **FSM alone is insufficient** because it cannot detect behavioral abuse or statistical outliers that follow legal rules. The Isolation Forest is designed specifically to detect this category.
- **Representative Patterns:**
  All 500 statistical anomalies in our dataset are characterized by **excessive payment-failure and retry loops (3 to 6 cycles)** that remain technically compliant with FSM rules, followed by either successful fulfillment or valid cancellation:
  - **Excessive Payment Retries Followed by Successful Fulfillment:**
    A customer or automated script fails payment and retries 3 to 6 times before finally confirming payment and completing fulfillment:
    `[FAIL_PAYMENT, RETRY_PAYMENT, FAIL_PAYMENT, RETRY_PAYMENT, FAIL_PAYMENT, RETRY_PAYMENT, CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]`
    *(FSM permits this cycle indefinitely; ML isolates the trace due to extreme sequence length $N \ge 10$, failure count $\ge 3$, and high retry ratio)*
  - **Excessive Payment Retries Followed by Order Abandonment/Cancellation:**
    A customer or bot fails and retries payment 3 to 6 times before abandoning the checkout by cancelling:
    `[FAIL_PAYMENT, RETRY_PAYMENT, FAIL_PAYMENT, RETRY_PAYMENT, FAIL_PAYMENT, RETRY_PAYMENT, CANCEL_ORDER]`
    *(Valid cancellation, but extreme number of retries before quitting: sequence length $N \ge 7$, failure count $\ge 3$, payment failure rate $= 1.0$)*

### 2.4 Taxonomy Comparison Matrix

| Anomaly Category | FSM Status | `is_ml_anomaly` Label | Target Detection Engine | Workflow Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `NORMAL` | Valid | `False` | Baseline Normal | Typical everyday purchase or clean cancellation. |
| `RULE_VIOLATION` | **Invalid** | `null` (N/A) | **FSM Engine** | System glitch, race condition, data corruption, or bypass. |
| `STATISTICAL_ANOMALY` | **Valid** | **`True`** | **ML Engine (Isolation Forest)** | Card brute-forcing, bot retry loops, excessive checkout friction. |

---

## 3. Dataset Structure & Schema

The dataset is persisted as a single tabular CSV file (`orders_workflow_dataset.csv`).

### 3.1 Field Definitions

| Column Name | Data Type | Role in Pipeline | Description / Definition |
| :--- | :--- | :--- | :--- |
| `trace_id` | `string` | Metadata / Key | Unique identifier for the order event trace (e.g., `TRACE_00001`). |
| `event_sequence` | `string` | **Raw Input** | Semicolon-delimited list of ordered events (e.g., `CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS`). |
| `sequence_length` | `integer` | Feature / Audit | Total count of events in the sequence ($N \ge 0$). |
| `is_rule_valid` | `boolean` | Ground-Truth Label | `True` if FSM validator evaluates all transitions as legal; `False` if any violation occurs. |
| `is_complete` | `boolean` | Ground-Truth Label | `True` if `is_rule_valid` is `True` and the sequence terminates in `DELIVERED`, `CANCELLED`, or `REFUNDED`. (Always `True` for generated `NORMAL` and `STATISTICAL_ANOMALY` traces; `False` for `RULE_VIOLATION` traces. Preserved for auditing and FSM evaluation; never fed to ML as an input feature). |
| `anomaly_category` | `string` | Multi-class Label | One of: `NORMAL`, `RULE_VIOLATION`, `STATISTICAL_ANOMALY`. |
| `is_ml_anomaly` | `boolean (nullable)` | Binary ML Label | Ground-truth flag for primary ML evaluation: `False` for `NORMAL`, `True` for `STATISTICAL_ANOMALY`, and `null` for `RULE_VIOLATION`. In CSV format, `null` is represented as an empty string between delimiters. |

### 3.2 Sample Rows (CSV Format)

Notice that for `RULE_VIOLATION` rows, `is_ml_anomaly` is rendered as an empty string (representing `null`):

```csv
trace_id,event_sequence,sequence_length,is_rule_valid,is_complete,anomaly_category,is_ml_anomaly
TRACE_00001,CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS,4,True,True,NORMAL,False
TRACE_00002,CANCEL_ORDER,1,True,True,NORMAL,False
TRACE_00003,FAIL_PAYMENT;RETRY_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS,6,True,True,NORMAL,False
TRACE_00004,CONFIRM_PAYMENT;SHIP_ITEMS;DELIVER_ITEMS,3,False,False,RULE_VIOLATION,
TRACE_00005,CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS;CANCEL_ORDER,5,False,False,RULE_VIOLATION,
TRACE_00006,CONFIRM_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS,3,False,False,RULE_VIOLATION,
TRACE_00007,FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS,10,True,True,STATISTICAL_ANOMALY,True
TRACE_00008,FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;CANCEL_ORDER,7,True,True,STATISTICAL_ANOMALY,True
```

### 3.3 Leakage Prevention: Features vs. Labels Separation

> [!CAUTION]
> **Data Leakage Risk:** Under no circumstances should label columns (`is_rule_valid`, `is_complete`, `anomaly_category`, `is_ml_anomaly`) or identification keys (`trace_id`) be supplied as inputs to the Isolation Forest model.
>
> The ML model consumes **strictly numerical features extracted from `event_sequence`** (e.g., event counts, n-grams, retry ratios). The labels are used **only** after model inference to compute evaluation metrics (Precision, Recall, F1, ROC-AUC).

---

## 4. Dataset Generation Methodology & Rules

To ensure realistic distributions, reproducible results, and deterministic ground truths, synthetic trace generation is governed by formal stochastic rules.

### 4.1 Generation Rules by Category

#### 1. Generating Normal Sequences (`NORMAL`)
Generated by simulating a random walk on the FSM transition graph starting at `ORDER_CREATED` with realistic branch probabilities. Every normal sequence proceeds until a terminal state (`DELIVERED`, `CANCELLED`, or `REFUNDED`) is reached.

**Crucial Constraint: Strict Single-Retry Limit**
The normal trace generator enforces a maximum of **ONE** payment retry across the entire trace. To prevent infinite loops or accidental multi-retry cycles (since `RETRY_PAYMENT` returns the order to `ORDER_CREATED`), the generator maintains an explicit counter: `retry_count = 0`.

- **At `ORDER_CREATED` (`retry_count == 0`):**
  - 85% probability: `CONFIRM_PAYMENT` $\to$ `PAID`
  - 10% probability: `CANCEL_ORDER` $\to$ `CANCELLED` (Terminal)
  - 5% probability: `FAIL_PAYMENT` $\to$ `PAYMENT_FAILED`
- **At `PAYMENT_FAILED`:**
  - 80% probability: `RETRY_PAYMENT` $\to$ `ORDER_CREATED` (increments `retry_count := 1`)
  - 20% probability: `CANCEL_ORDER` $\to$ `CANCELLED` (Terminal)
- **At `ORDER_CREATED` (`retry_count >= 1`):**
  - The branch to `FAIL_PAYMENT` is disabled (probability set to 0.0).
  - 90% probability: `CONFIRM_PAYMENT` $\to$ `PAID`
  - 10% probability: `CANCEL_ORDER` $\to$ `CANCELLED` (Terminal)
- **At `PAID`:**
  - 95% probability: `PACK_ITEMS` $\to$ `PACKED`
  - 5% probability: `REFUND_ORDER` $\to$ `REFUNDED` (Terminal)
- **At `PACKED`:**
  - 97% probability: `SHIP_ITEMS` $\to$ `SHIPPED`
  - 3% probability: `REFUND_ORDER` $\to$ `REFUNDED` (Terminal)
- **At `SHIPPED`:**
  - 100% probability: `DELIVER_ITEMS` $\to$ `DELIVERED` (Terminal)

This strictly bounds retries to $\le 1$ for normal traces while generating only valid transitions according to the FSM.

#### 2. Generating Rule-Invalid Sequences (`RULE_VIOLATION`)
Generated by taking a valid base sequence and applying one of six deterministic **mutation operators**:
1. **Drop Prerequisite (`DROP`):** Remove an essential intermediate event (e.g., remove `PACK_ITEMS`, turning `PAID -> PACKED -> SHIPPED` into `PAID -> SHIPPED`).
2. **Swap Order (`SWAP`):** Invert two adjacent steps (e.g., `SHIP_ITEMS` before `PACK_ITEMS`).
3. **Premature Action (`PREMATURE`):** Insert a late-stage event at the start (e.g., `DELIVER_ITEMS` as the first event).
4. **Zombie Mutation (`POST_TERMINAL`):** Append an action after reaching a terminal state (e.g., `CANCEL_ORDER` after `DELIVERED`).
5. **Duplicate Event (`DUPLICATE`):** Repeat a non-looping event in place (e.g., `CONFIRM_PAYMENT` twice).
6. **Inject Unknown Symbol (`UNKNOWN`):** Insert an out-of-alphabet event (e.g., `HACK_CHECKOUT` or `BYPASS_GATEWAY`).

Every generated invalid sequence is verified through [`validate_trace()`](file:///c:/Users/SUPRATEEK/TOC_PROJECT/backend/fsm/validator.py) to confirm that `is_rule_valid == False`.

#### 3. Generating Statistically Unusual Sequences (`STATISTICAL_ANOMALY`)
All 500 statistical anomalies are generated using valid FSM transitions, driving the state machine into extreme behavioral loops that deviate drastically from normal user behavior while remaining 100% compliant with FSM rules:

1. **Loop Generation Process:**
   - Sample the number of failure-retry cycles $k \in \{3, 4, 5, 6\}$ uniformly at random.
   - For each of the $k$ cycles, append the pair `FAIL_PAYMENT` followed by `RETRY_PAYMENT`.
   - After $k$ cycles, the trace has accumulated $2k$ events (between 6 and 12 events) and the order has returned to `ORDER_CREATED`.

2. **Terminal Resolution Path:**
   From `ORDER_CREATED`, the sequence proceeds to completion via one of two valid resolution paths:
   - **Successful Fulfillment:** `CONFIRM_PAYMENT -> PACK_ITEMS -> SHIP_ITEMS -> DELIVER_ITEMS` (terminating at `DELIVERED`). Total sequence length: $2k + 4$ (ranging from 10 to 16 events).
   - **Valid Cancellation:** `CANCEL_ORDER` (terminating at `CANCELLED`). Total sequence length: $2k + 1$ (ranging from 7 to 13 events).

3. **Guaranteed Label Disjointness (Zero Label Contradiction):**
   - In `NORMAL` generation, the random-walk generator strictly enforces `retry_count <= 1`. Thus, every normal trace contains at most one `FAIL_PAYMENT` and at most one `RETRY_PAYMENT`.
   - In `STATISTICAL_ANOMALY` generation, every trace contains $k \ge 3$ cycles, producing at least three `FAIL_PAYMENT` and three `RETRY_PAYMENT` events.
   - Consequently, the set of event sequences in `NORMAL` and the set of event sequences in `STATISTICAL_ANOMALY` are strictly disjoint ($\mathcal{S}_{\text{NORMAL}} \cap \mathcal{S}_{\text{STATISTICAL\_ANOMALY}} = \emptyset$). No identical event sequence can ever be assigned conflicting labels.

4. **Validation:**
   Every generated statistical anomaly is verified through [`validate_trace()`](file:///c:/Users/SUPRATEEK/TOC_PROJECT/backend/fsm/validator.py) to confirm that `is_rule_valid == True` and `is_complete == True`.

---

### 4.2 Dataset Size & Partition Breakdown

The total dataset size is exactly **5,000 traces**, allocated as follows:

| Category | Total Count | Proportion | Partition Role | Count in Partition |
| :--- | :---: | :---: | :--- | :---: |
| **NORMAL** | **3,500** | 70.0% | **Isolation Forest Training** | **2,800** |
| | | | **Primary ML Testing (Negative Class)** | **700** |
| **RULE_VIOLATION** | **1,000** | 20.0% | **FSM Benchmark Testing** | **1,000** |
| **STATISTICAL_ANOMALY** | **500** | 10.0% | **Primary ML Testing (Positive Class)** | **500** |
| **Total** | **5,000** | **100.0%** | **Complete Dataset** | **5,000** |

**Verification of Sums:**
- Training set: $2,800$ (Normal)
- Test pool: $700\text{ (Normal)} + 500\text{ (Statistical Anomaly)} + 1,000\text{ (Rule Violation)} = 2,200$
- Total: $2,800 + 2,200 = 5,000$ traces.

---

### 4.3 Reproducibility & Random Seed

To guarantee identical results across runs, machines, and environments:
- Generation relies on a master seed: `RANDOM_SEED = 42`.
- Both Python's built-in `random.seed(42)` and NumPy's `numpy.random.seed(42)` must be initialized at the start of dataset generation.
- Running the generator script multiple times will produce the exact same byte-for-byte CSV output.

---

### 4.4 Handling Duplicate Sequences

In typical order workflows, many transactions naturally follow the exact same event progression (for example, the standard 4-step fulfillment sequence `[CONFIRM_PAYMENT, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS]`).
- **In the Dataset:** Duplicate `event_sequence` strings are **expected and preserved** across normal traces. Each order is distinguished by its unique `trace_id`. These repeated sequences reflect common baseline operational behavior.
- **In ML Training:** In an Isolation Forest, trees are constructed through random axis-aligned splits on feature values. Because common baseline traces share standard feature values (such as normal sequence lengths and zero retry counts), their feature points typically require deeper recursive partitioning (greater average tree depth) to isolate. In contrast, instances with unusual feature combinations (such as high failure counts or elevated retry ratios) can be separated in fewer splits and may be isolated more easily by the trees. Preserving duplicate normal sequences accurately mirrors realistic baseline volume, though it does not by itself guarantee anomaly detection for every arbitrary variation.

---

## 5. Training & Evaluation Strategy for Isolation Forest

### 5.1 Training Set vs. Test Set Partitioning

The dataset is partitioned into clean functional subsets:

```text
5,000 Total Generated Traces
│
├── Training Set (2,800 traces)
│   └── 2,800 NORMAL traces (100% clean normal baseline)
│
└── Test Pool (2,200 traces)
    ├── Primary ML Evaluation Set (1,200 traces)
    │   ├── 700 NORMAL traces (is_ml_anomaly = False)
    │   └── 500 STATISTICAL_ANOMALY traces (is_ml_anomaly = True)
    │
    └── FSM Evaluation Set (1,000 traces)
        └── 1,000 RULE_VIOLATION traces (is_rule_valid = False)
```

#### Why Rule Violations are Excluded from Primary ML Evaluation:
Rule violations represent structural syntax errors that are already 100% caught by the deterministic FSM engine. Including them as positive examples in the primary ML benchmark would artificially inflate ML metrics on trivial structural breaks (e.g. unknown symbols) rather than measuring its ability to detect **subtle, structurally legal behavioral anomalies**.

The ML model may still produce exploratory anomaly scores for rule-invalid traces, but they are excluded from the primary precision, recall, and ROC-AUC metrics.

---

### 5.2 Isolation Forest Configuration

The model is instantiated as:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)
```

#### Why `contamination="auto"` is Appropriate:
- **Clean Normal Training:** Because the training set consists of 2,800 purely normal traces without injected outliers, forcing an arbitrary numerical contamination (such as `contamination=0.1`) would artificially force the algorithm to mark 10% of genuine normal transactions as anomalies.
- **Paper Baseline:** Setting `contamination="auto"` uses the original threshold formulation defined by Liu et al. (the Isolation Forest authors), setting the decision threshold offset without forcing an artificial outlier fraction on clean baseline data.

---

### 5.3 Anomaly Scoring & ROC-AUC Calculation

In Scikit-learn, `model.decision_function(X)` is defined such that:
- Inliers (normal points) receive **positive** scores.
- Outliers (anomalies) receive **negative** scores.

However, standard ROC-AUC calculation (`roc_auc_score(y_true, y_score)`) requires that **higher scores correspond to greater likelihood of being the positive class (`is_ml_anomaly = True`)**.

Therefore, the anomaly score must be negated:

```python
# Invert decision function so higher values indicate greater anomaly likelihood
anomaly_scores = -model.decision_function(X_test)
```

This ensures that:
- Extreme retry anomalies have high positive anomaly scores (e.g., $+0.25$).
- Standard happy path normal traces have low negative anomaly scores (e.g., $-0.15$).
- The ROC-AUC metric correctly evaluates ranking discrimination.

---

### 5.4 Leakage Prevention Protocol

1. **Split Prior to Transformation:** The dataset is split into `train_traces` and `test_traces` using `trace_id` before feature extraction and normalization.
2. **Fit on Train, Transform on Test:** Any feature scalers (e.g., `StandardScaler` or `MinMaxScaler`) are fitted **only** on `X_train` and then applied to `X_test` to prevent distribution leakage.
3. **No Label Leakage:** Labels (`is_rule_valid`, `is_complete`, `anomaly_category`, `is_ml_anomaly`) are never passed to feature transformers or model inputs.

---

### 5.5 Realistic Evaluation Metrics

Isolation Forest is an unsupervised tree-based heuristic based on random partitioning; it does not guarantee 100% detection of every anomaly, nor does it output calibrated probabilities. Evaluation on the 1,200-trace primary ML test set uses:
- **Precision:** $\frac{TP}{TP + FP}$ — When the model predicts an anomaly (`predict == -1`), how often is it a true statistical anomaly?
- **Recall:** $\frac{TP}{TP + FN}$ — What fraction of the 500 statistical anomalies were flagged?
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC Score:** Area under the ROC curve using `anomaly_scores = -decision_function(X_test)`.

---

## 6. Feature Extraction Engineering

The Isolation Forest algorithm requires a fixed-dimensional numerical feature matrix:
$$X \in \mathbb{R}^{N \times D}$$

Every raw `event_sequence` is converted into a vector of $D = 14$ engineered features:

### 6.1 Feature Dictionary

| Feature Index | Feature Name | Type | Formula / Logic | Intuition / Anomaly Signal |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `seq_length` | `int` | $N = \text{len}(\text{trace})$ | Statistical anomalies with retry loops have significantly higher length ($N \ge 7$). |
| 2 | `count_CONFIRM_PAYMENT` | `int` | Count of `CONFIRM_PAYMENT` | Normal traces have 0 or 1; duplicates have $\ge 2$. |
| 3 | `count_FAIL_PAYMENT` | `int` | Count of `FAIL_PAYMENT` | Normal traces have 0 or 1; retry anomalies have $\ge 3$. |
| 4 | `count_RETRY_PAYMENT` | `int` | Count of `RETRY_PAYMENT` | Direct signal of customer friction or bot retries ($\ge 3$ in retry anomalies). |
| 5 | `count_CANCEL_ORDER` | `int` | Count of `CANCEL_ORDER` | Normal traces have 0 or 1. |
| 6 | `count_PACK_ITEMS` | `int` | Count of `PACK_ITEMS` | Normal traces have 0 or 1. |
| 7 | `count_SHIP_ITEMS` | `int` | Count of `SHIP_ITEMS` | Normal traces have 0 or 1. |
| 8 | `count_DELIVER_ITEMS` | `int` | Count of `DELIVER_ITEMS` | Normal traces have 0 or 1. |
| 9 | `count_REFUND_ORDER` | `int` | Count of `REFUND_ORDER` | Identifies cancellation/refund pathways. |
| 10 | `retry_ratio` | `float` | $\frac{\text{count\_RETRY\_PAYMENT}}{\max(1, N)}$ | Proportion of sequence spent retrying failed operations. |
| 11 | `payment_failure_rate` | `float` | $\frac{\text{count\_FAIL\_PAYMENT}}{\max(1, \text{count\_CONFIRM\_PAYMENT} + \text{count\_FAIL\_PAYMENT})}$ | Ratio of payment attempts that failed ($0.0$ in happy path, $\ge 0.75$ in retry anomalies). |
| 12 | `bigram_FAIL_to_RETRY` | `int` | Count of adjacent `FAIL_PAYMENT -> RETRY_PAYMENT` pairs | Measures frequency of retry cycle oscillations ($\ge 3$ in retry anomalies). |
| 13 | `bigram_PACK_to_SHIP` | `int` | Count of adjacent `PACK_ITEMS -> SHIP_ITEMS` pairs | Standard progression feature. |
| 14 | `unique_event_count` | `int` | Number of distinct event types in trace | Normal happy path = 4; simple cancel = 1; retry anomaly followed by fulfillment = 6; retry anomaly followed by cancellation = 3. |

> [!NOTE]
> **Omission of `is_terminal_reached`:** Because all traces in the primary ML dataset (both normal and statistical anomaly) reach terminal states, `is_terminal_reached` would equal $1$ for 100% of rows, providing zero variance. It has therefore been omitted from the ML feature set. `is_complete` remains in the CSV schema for auditing and FSM benchmarking.

### 6.2 Feature Vector Transformation Example

Consider two traces:
1. **Normal Happy Path:** `CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS`
2. **Statistical Anomaly (3 Retries):** `FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;FAIL_PAYMENT;RETRY_PAYMENT;CONFIRM_PAYMENT;PACK_ITEMS;SHIP_ITEMS;DELIVER_ITEMS`

```text
Feature                      Normal Trace Vector     Statistical Anomaly Vector
───────────────────────────────────────────────────────────────────────────────
1.  seq_length                        4                         10
2.  count_CONFIRM_PAYMENT             1                          1
3.  count_FAIL_PAYMENT                0                          3
4.  count_RETRY_PAYMENT               0                          3
5.  count_CANCEL_ORDER                0                          0
6.  count_PACK_ITEMS                  1                          1
7.  count_SHIP_ITEMS                  1                          1
8.  count_DELIVER_ITEMS               1                          1
9.  count_REFUND_ORDER                0                          0
10. retry_ratio                      0.00                       0.30
11. payment_failure_rate             0.00                       0.75
12. bigram_FAIL_to_RETRY              0                          3
13. bigram_PACK_to_SHIP               1                          1
14. unique_event_count                4                          6
───────────────────────────────────────────────────────────────────────────────
Expected Isolation Depth        Requires deeper splits     Isolated in fewer splits
                                (typical baseline)         (unusual feature profile)
```

In the feature space, the normal trace exhibits feature values aligned with typical baseline executions, meaning an Isolation Forest requires more recursive random cuts to isolate it. Conversely, the statistical anomaly possesses extreme feature values (such as `seq_length = 10` and `count_FAIL_PAYMENT = 3`), allowing the random binary trees to isolate the instance with shorter average path lengths.

---

## 7. Next Implementation Steps (Phase 4)

Once this specification is approved, implementation will proceed in two organized steps:
1. **Implement Generator (`backend/ml/dataset/generator.py`):**
   - Implements stochastic generator functions respecting the single-retry limit for normal traces and generating the exact 5,000-trace split.
   - Saves `orders_workflow_dataset.csv` into `backend/ml/dataset/data/`.
   - Incorporates random seed reproducibility (`seed=42`) and automated validation against `validate_trace()`.
2. **Implement Feature Extractor (`backend/ml/feature_extraction/extractor.py`):**
   - Transforms event sequence strings into the 14-dimensional numeric feature matrix.
   - Splits traces into 2,800 training and 2,200 test rows, avoiding data leakage.
