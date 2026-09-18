# GridWise LLM — Smart Campus Energy Optimization Service

[![Evaluation Compatibility](https://img.shields.io/badge/BUP%20CSE%20Fest%202026-Online%20Preliminary-blue.svg)](https://fest.bupcopc.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Optimization](https://img.shields.io/badge/Solver-HiGHS%20Linear%20Programming-orange.svg)](https://scipy.org/)
[![Docker Fallback](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)

GridWise LLM is an autonomous, production-grade HTTP API service designed for the **BUP CSE Fest 2026 Hackathon (Smart Campus Energy Optimization Challenge)** in association with **Poridhi.io**.

It bridges natural-language campus operational notices with rigorous mathematical energy scheduling using a verified 4-stage pipeline: **LLM Directive Interpretation** $\rightarrow$ **Deterministic Guardrails** $\rightarrow$ **HiGHS Linear Programming Optimizer** $\rightarrow$ **Strict JSON Formatter**.

---

## 1. System Architecture

```
                                  [Campus Scenario Request]
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
             [Operator Notes]                                [24h Demand, Solar, Tariffs,
            (1-3 natural text)                                 Battery Capacity & Limits]
                     │                                                 │
                     ▼                                                 │
        ┌─────────────────────────┐                                    │
        │ Stage 1: LLM Interpreter│ (Google Gemini 2.5/Flash-Lite)     │
        └────────────┬────────────┘                                    │
                     ▼                                                 │
        ┌─────────────────────────┐                                    │
        │Stage 2: Guardrail Engine│ (Deterministic bounds, range,      │
        │     & Safe Fallback     │  format, & schema validation)      │
        └────────────┬────────────┘                                    │
                     │                                                 │
                     ▼ (Validated Directives)                          ▼
        ┌──────────────────────────────────────────────────────────────┴─┐
        │             Stage 3: Mathematical Optimizer (HiGHS LP)         │
        │                                                                │
        │  • Effective solar calculation after solar_reduction           │
        │  • Minimum battery reserve bound adjustments                   │
        │  • No-charge and no-discharge operational windows              │
        │  • Max-grid intake feeder/transformer limits                   │
        │  • Hourly energy balance: Grid + Solar + Disch = Demand + Chg  │
        │  • Battery state transition: E[h] = E[h-1] + Chg - Disch       │
        │  • End-of-day battery neutrality: E[23] == E_initial           │
        │  • Objective: MINIMIZE total_cost_bdt = SUM(grid * tariff)     │
        └──────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────────┐
        │                 Stage 4: Canonical Response Formatter          │
        │   • Strict JSON contract (scenario_id, directive_interpretation│
        │     hourly_plan[24], totals, recalculation, plan_summary)      │
        └──────────────────────────────┬─────────────────────────────────┘
                                       │
                                       ▼
                             [HTTP 200 JSON Response]
```

---

## 2. API Contract & Endpoints

### `GET /health`
Readiness check for the automated judging harness.
* **HTTP Status**: `200 OK`
* **Response**:
```json
{
  "status": "ok"
}
```

### `POST /optimize-energy`
Main optimization endpoint accepting a 24-hour scenario plus operator notes.
* **Request Shape**:
```json
{
  "scenario_id": "GRID-101",
  "operator_notes": [
    "Solar output will drop to about 20% from 1 PM to 3 PM.",
    "Do not charge the battery between 2 PM and 4 PM.",
    "The cafeteria menu changes tomorrow."
  ],
  "hours": [
    {"hour": 0, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 7},
    "... 22 more entries ...",
    {"hour": 23, "demand_kwh": 200, "solar_kwh": 0, "tariff_bdt_per_kwh": 9}
  ],
  "battery": {
    "capacity_kwh": 500,
    "initial_energy_kwh": 200,
    "minimum_energy_kwh": 50,
    "max_charge_kwh_per_hour": 100,
    "max_discharge_kwh_per_hour": 100
  }
}
```

* **Response Shape**:
```json
{
  "scenario_id": "GRID-101",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
      "explanation": "Solar reduction applied."
    },
    {
      "note_index": 1,
      "applies": true,
      "directive_type": "no_charge_window",
      "structured_adjustment": {"hours": [14, 15]},
      "explanation": "Battery charging disabled."
    },
    {
      "note_index": 2,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "Note does not alter energy schedule."
    }
  ],
  "hourly_plan": [
    {
      "hour": 0,
      "grid_kwh": 180.0,
      "solar_used_kwh": 0.0,
      "battery_action": "idle",
      "battery_kwh": 0.0,
      "battery_energy_after_kwh": 200.0
    }
  ],
  "total_grid_kwh": 3120.0,
  "total_cost_bdt": 24960.0,
  "peak_grid_kwh": 250.0,
  "plan_summary": "Optimized 24-hour schedule..."
}
```

---

## 3. Local Quickstart (Zero-Friction Reproducibility)

### Prerequisites
* Python 3.10+ (Tested up to Python 3.14)
* `pip`

### Step 1: Clone & Setup Environment
```bash
git clone https://github.com/shaymimran45/bup-cse-fest-gridwise.git
cd bup-cse-fest-gridwise

# Optional virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` and provide your Google Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
PORT=8000
HOST=0.0.0.0
```

### Step 3: Run the Service
```bash
python main.py
# or: uvicorn main.py:app --host 0.0.0.0 --port 8000
```

### Step 4: Verify Health Endpoint
```bash
curl -X GET http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Step 5: Test with Public Sample Case
```bash
python test_samples.py
```
This automatically exercises all 10 public test cases, validates directives, physics constraints, battery neutrality, and checks costs against the reference benchmarks within the 0.01 BDT official tolerance.

---

## 4. Docker Fallback Execution

The repository includes a production-grade multi-stage Dockerfile that runs as a non-root user and exposes port 8000.

### Build Container Image
```bash
docker build -t gridwise-llm:latest .
```

### Run Container
```bash
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY="your_api_key_here" \
  -e PORT=8000 \
  --name gridwise-service \
  gridwise-llm:latest
```

### Test Running Container
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

---

## 5. Implementation Details & Technologies

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **Language Model** | Google Gemini (`gemini-2.5-flash-lite` / `gemini-3.6-flash`) | Low p95 latency (<2s), high structured JSON compliance, handles paraphrasing seamlessly. |
| **Deterministic Guardrails** | Custom Python Engine (`guardrails.py`) | Enforces 100% adherence to Section 08 rules (unique ascending hours 0-23, bounded numeric values, applies semantics). |
| **Optimizer / Solver** | HiGHS Linear Programming (`scipy.optimize.linprog`) | Solves 120-variable LP in <2ms with mathematical optimality, zero floating drift, and guaranteed energy balance. |
| **Web Framework** | FastAPI + Pydantic v2 | High throughput, asynchronous non-blocking I/O, strict automated schema validation. |

---

## 6. Security & Secret Handling
* **No hardcoded secrets**: API keys and tokens are loaded strictly via environment variables.
* **Safe logging**: Prompts, API responses, and logs never dump credential tokens or sensitive stack traces.
* **Safe failures**: Malformed inputs return HTTP 400 with structured validation errors, preventing 500 crashes.

---

## 7. Known Limitations & Edge Cases
* Time intervals in the Problem Statement are start-inclusive and end-exclusive (e.g. `1 PM to 3 PM` $\rightarrow$ `[13, 14]`). Minutes are rounded to nearest whole hours.
* Floating-point numbers are rounded to 4 decimal places in response JSON and meet the canonical $\pm 0.01$ kWh / BDT tolerance rule.

