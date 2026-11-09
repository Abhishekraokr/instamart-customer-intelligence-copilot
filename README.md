# Instamart Customer Intelligence & Support Copilot

An enterprise-grade customer intelligence and automated support copilot for quick commerce operations, built strictly offline and deterministically across four core components:
- **Part A**: Data Pipeline, Relational Storage & SQL Analytics
- **Part B**: Customer Segmentation & Lapse Prediction
- **Part C**: Customer Support Ticket Triage via LLM
- **Part D**: Policy RAG & ReAct Refund Copilot

---

## 1. System Architecture

The copilot integrates data engineering, predictive machine learning, and agentic LLM workflows into an autonomous quick-commerce pipeline:

```
+---------------------------------------------------------------------------------------+
|                                    RAW DATA SOURCES                                   |
|   orders.csv (13,084) | customers.csv (1,200) | products.csv (40) | tickets.csv (164) |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                    PART A: DATA CLEANING & RELATIONAL STORAGE (SQLite)                |
|  - Deduplication (latest order_ts) & status coercion (pending -> cancelled)           |
|  - Formula-based amount imputation: quantity * unit_price * (1 - discount_pct/100)    |
|  - City & gender standardisation; median age imputation; completed delivery time logic|
|  - Output: data/clean/*.csv  &  data/instamart.db (indexed tables)                    |
+---------------------+-------------------------------------+---------------------------+
                      |                                     |
                      v                                     v
+---------------------------------------+ +---------------------------------------------+
|        PART A: SQL ANALYTICS          | |       PART B: SEGMENTATION & LAPSE          |
|  - Executive queries Q1 - Q7          | |  - RFM Feature Engineering (snapshot date)  |
|  - Revenue cross-tabulation by city   | |  - K-Means (K=4, silhouette = 0.526)        |
|  - Cumulative revenue over 183 days   | |  - Premium Segment 1 flag (INR 7,129.70 avg)|
|  - Output: sql/queries.sql            | |  - Leak-free Random Forest lapse prediction |
|  - Plots: outputs/plots/*.png         | |  - Outputs: outputs/customer_segments.csv   |
|                                       | |             outputs/lapse_model.joblib      |
|                                       | |             outputs/retention_list.csv      |
+---------------------------------------+ +---------------------------------------------+
                      |                                     |
                      v                                     v
+---------------------------------------+ +---------------------------------------------+
|    PART C: SUPPORT TICKET TRIAGE      | |     PART D: POLICY RAG & REACT COPILOT      |
|  - Regex PII redaction (278 redacted) | |  - 15 semantic policy chunks (10-w overlap) |
|  - Guardrails (empty / long tickets)  | |  - ReAct agent with 5 deterministic tools   |
|  - Few-shot department routing (100%) | |  - Idempotent refund ledger & safety limits |
|  - Token budget demo (6,000 ceiling)  | |  - State checkpoint persistence & resume    |
|  - Output: outputs/triage_results.csv | |  - Outputs: outputs/agent_runs.jsonl        |
|            outputs/triage_budget.csv  | |             outputs/refund_ledger.jsonl     |
|                                       | |             state/*.json                    |
+---------------------------------------+ +---------------------------------------------+
```

### Component Details
1. **Data Ingestion & SQL (Part A)**: Cleans 13,084 raw orders into 12,309 unique orders, validates relational integrity in SQLite (`data/instamart.db`), executes analytical queries Q1–Q7 (`sql/queries.sql`), and generates exploratory plots (`outputs/plots/`).
2. **Segmentation & Lapse Prediction (Part B)**: Clusters 1,175 active customers into 4 RFM segments (flagging Segment 1 as Premium VIP), and trains a leak-free Random Forest model predicting 30-day lapse with class-balanced evaluation (`outputs/lapse_model.joblib`).
3. **Support Ticket Triage (Part C)**: Sanitizes PII across 164 customer support tickets, handles simulated transient API failures (`503 server busy`) with exponential backoff, and routes tickets into 6 categories with priority tags and draft responses.
4. **Policy RAG & ReAct Copilot (Part D)**: Evaluates policy document retrieval with chunk overlap, runs 7 distinct support scenarios (S1–S7) through an autonomous ReAct loop, enforces safety limits, and verifies crash recovery via JSON state checkpoints (`state/`).

---

## 2. Running Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.12)
- Git

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/Abhishekraokr/instamart-customer-intelligence-copilot.git
cd instamart-customer-intelligence-copilot

# 2. Create and activate virtual environment
python -m venv venv

# On Linux / WSL:
source venv/bin/activate

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: Create local environment configuration
cp .env.example .env
```

### Run Everything End-to-End
Run the entire pipeline across all four parts in a single automated command:
```bash
python run_all.py
```
*Expected runtime: ~35 seconds on modern hardware.*

### Run Individual Modular Components
Each pipeline stage can also be executed independently:

```bash
# Part A: Data cleaning & SQLite database build
python src/clean_data.py

# Part A: SQL analytics queries (Q1 - Q7) & statistics
python src/sql_analysis.py

# Part A: Generate exploratory .png plots via plt.savefig()
python src/generate_plots.py

# Part B: Customer RFM segmentation & Random Forest lapse model
python src/segments_and_lapse.py

# Part C: Customer support ticket PII redaction & LLM triage
python src/ticket_triage.py

# Part D: Policy RAG evaluation & ReAct agent scenarios (S1 - S7)
python src/policy_rag_and_agent.py
```

---

## 3. Environment Configuration & Security

This project runs **100% offline and deterministically** using the provided `mock_llm.py` stand-in. No paid external API keys, network calls, or secrets are required.

- `.gitignore` automatically excludes `.env`, `venv/`, `__pycache__/`, and local runtime files.
- `.env.example` provides safe defaults:
  ```env
  MOCK_LLM_SEED=42
  LOG_LEVEL=INFO
  ```

---

## 4. Repository Structure

```
instamart-customer-intelligence-copilot/
├── generate_data.py            # Unmodified synthetic dataset generator
├── mock_llm.py                 # Unmodified offline mock LLM
├── requirements.txt            # Pinned dependencies
├── .gitignore                  # Excludes venv, .env, __pycache__, temporary artifacts
├── .env.example                # Safe environment variable template (No secrets policy)
├── README.md                   # System architecture and running instructions
├── run_all.py                  # End-to-end master pipeline orchestrator
│
├── data/
│   ├── clean/                  # Cleaned and validated CSV datasets
│   │   ├── customers.csv       # Cleaned customer demographics (1,200 rows)
│   │   ├── products.csv        # Product catalog (40 items across 8 categories)
│   │   └── orders.csv          # Cleaned order transactions (12,309 rows)
│   ├── instamart.db            # Indexed relational SQLite database
│   ├── policies/               # Instamart operating policy documents
│   │   ├── delivery_policy.txt
│   │   ├── escalation_policy.txt
│   │   ├── payment_policy.txt
│   │   ├── privacy_policy.txt
│   │   └── refund_policy.txt
│   └── raw/                    # Raw generated datasets
│
├── sql/
│   └── queries.sql             # SQL business analytics queries (Q1 - Q7)
│
├── src/                        # Modular pipeline source code
│   ├── clean_data.py           # Part A: Data cleaning, validation & SQLite ingestion
│   ├── sql_analysis.py         # Part A: SQL query execution & EDA statistics
│   ├── generate_plots.py       # Part A: Saved .png plots generated via plt.savefig()
│   ├── segments_and_lapse.py   # Part B: RFM clustering, PCA & Random Forest modeling
│   ├── ticket_triage.py        # Part C: Input guardrails, PII masking & LLM triage
│   └── policy_rag_and_agent.py # Part D: Policy chunking RAG & ReAct refund agent
│
├── outputs/                    # Pipeline outputs and persistent artifacts
│   ├── customer_segments.csv   # RFM segments with is_premium classification
│   ├── lapse_model.joblib      # Serialized Random Forest model bundle
│   ├── retention_list.csv      # High-priority retention action list
│   ├── triage_results.csv      # Final classified and drafted support tickets
│   ├── triage_budget_demo.csv  # Token ceiling budget exhaustion demonstration
│   ├── agent_runs.jsonl        # ReAct execution trace logs across scenarios S1 - S7
│   ├── refund_ledger.jsonl     # Idempotent financial refund transaction log
│   └── plots/                  # Visual exploratory data analysis plots (.png)
│       ├── order_amount_distribution.png
│       ├── daily_revenue_cumulative.png
│       ├── city_spend_breakdown.png
│       └── delivery_time_distribution.png
│
└── state/                      # Part D: Execution checkpoints & crash recovery snapshots
    └── *.json
```
