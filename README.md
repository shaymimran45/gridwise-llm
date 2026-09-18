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

## 3. Mathematical Formulation

The energy scheduler is solved as a **Linear Program (LP)** with `scipy.optimize.linprog(method="highs")`. The solver guarantees a globally optimal, mathematically proven minimum-cost schedule that satisfies all operator directives as hard constraints.

### 3.1 Decision Variables

For each hour $h \in \{0, 1, \ldots, 23\}$ we introduce five non-negative decision variables (total: $5 \times 24 = 120$ variables):

| Symbol | Index range | Meaning | Units |
| :--- | :--- | :--- | :--- |
| $g_h$ | $0\text{–}23$ | Grid import | kWh |
| $s_h$ | $24\text{–}47$ | Solar energy consumed | kWh |
| $c_h$ | $48\text{–}71$ | Battery charge rate | kWh/h |
| $d_h$ | $72\text{–}95$ | Battery discharge rate | kWh/h |
| $E_h$ | $96\text{–}119$ | Battery energy **after** hour $h$ | kWh |

### 3.2 Objective Function

**Minimize total operational cost in BDT:**

$$\min_{g, s, c, d, E} \; Z \;=\; \sum_{h=0}^{23} \tau_h \cdot g_h \;+\; \varepsilon \sum_{h=0}^{23} \left( c_h + d_h \right)$$

where:
* $\tau_h$ is the per-kWh grid tariff at hour $h$ (BDT/kWh).
* $\varepsilon = 10^{-7}$ is a tie-breaker coefficient that discourages gratuitous battery cycling without altering the optimal cost value (kept 7 orders of magnitude below the smallest realistic tariff so it cannot distort the schedule).

### 3.3 Equality Constraints ($A_{eq}\, x = b_{eq}$, 49 rows)

**(i) Hourly energy balance** — for each $h \in [0, 23]$:

$$g_h + s_h + d_h - c_h = D_h$$

where $D_h$ is the campus demand at hour $h$.

**(ii) Battery state transition** — for $h = 0$:

$$E_0 - c_0 + d_0 = E_{\text{init}}$$

and for $h \in [1, 23]$:

$$E_h - E_{h-1} - c_h + d_h = 0$$

**(iii) End-of-day neutrality** — battery returns to its initial state:

$$E_{23} = E_{\text{init}}$$

### 3.4 Variable Bounds

For each hour $h \in [0, 23]$:

$$
\begin{aligned}
0 \le g_h &\le G_h^{\max} \\
0 \le s_h &\le S_h^{\text{eff}} \\
0 \le c_h &\le C_h^{\max} \\
0 \le d_h &\le D_h^{\max} \\
R_h^{\min} \le E_h &\le C^{\text{bat}}
\end{aligned}
$$

where (in the unconstrained case):

$$
G_h^{\max} = +\infty, \quad S_h^{\text{eff}} = S_h, \quad C_h^{\max} = c^{\text{rate}}, \quad D_h^{\max} = d^{\text{rate}}, \quad R_h^{\min} = E_{\min}
$$

and $C^{\text{bat}}$, $c^{\text{rate}}$, $d^{\text{rate}}$, $E_{\min}$, $E_{\text{init}}$ are the battery spec values from the request.

### 3.5 Operator Directives as Linear Constraints

The guardrail layer rewrites each natural-language note into a structured adjustment that tightens the bounds above. Every directive maps cleanly to a linear inequality (already represented by upper/lower bounds, so no extra rows are added).

| Directive type | Bound it modifies | Mathematical effect |
| :--- | :--- | :--- |
| `solar_reduction` with factor $f$ | $S_h^{\text{eff}} \leftarrow S_h^{\text{eff}} \cdot f$ | Caps usable solar at hour $h$ |
| `solar_outage` | $S_h^{\text{eff}} \leftarrow 0$ | Forces $s_h = 0$ |
| `minimum_battery_reserve` $r$ | $R_h^{\min} \leftarrow \max(R_h^{\min}, r)$ | Lower bound on $E_h$ |
| `no_charge_window` | $C_h^{\max} \leftarrow 0$ | Forces $c_h = 0$ |
| `no_discharge_window` | $D_h^{\max} \leftarrow 0$ | Forces $d_h = 0$ |
| `max_grid_window` $M$ | $G_h^{\max} \leftarrow \min(G_h^{\max}, M)$ | Caps grid import |

**Solar factor extraction** (closed-form, see `_extract_solar_factor`):

$$
f_{\text{reduce}} \;=\; \mathrm{clamp}\!\left(1 - \frac{p}{100},\; 0,\; 1\right), \qquad f_{\text{absolute}} \;=\; \mathrm{clamp}\!\left(\frac{p}{100},\; 0,\; 1\right)
$$

where $p$ is the percentage parsed from the note (e.g. *"drop by 50%"* → $p=50$ → $f = 0.5$; *"to 20%"* → $p=20$ → $f = 0.2$; *"offline"* → $f = 0$).

**12-hour time conversion** (see `_to_24h`):

$$
h_{24} \;=\;
\begin{cases}
(h \bmod 12) + 12 & \text{if pm and } h < 12 \\[4pt]
0 & \text{if am and } h = 12 \\[4pt]
h & \text{if am and } h < 12 \quad \text{or no meridiem}
\end{cases}
$$

**Guardrail clamping** (see `validate_and_guardrail_directives`):

$$
f \in [0,\, 1], \qquad R_h^{\min} \in [0,\, C^{\text{bat}}], \qquad G_h^{\max} \ge 0
$$

### 3.6 LP Size and Solver Performance

* **Variables:** $n = 120$
* **Equality constraints:** $m_{eq} = 49$ (24 energy balances + 24 battery transitions + 1 end-of-day neutrality)
* **Inequality constraints:** none (all bounds are box constraints encoded in `bounds`)
* **Solver:** HiGHS sparse simplex / IPM
* **Typical solve time:** $< 2\,\text{ms}$ on a single core
* **Numerical guarantees:** global optimum, energy balance exact to solver tolerance, end-of-day battery neutrality exact.

### 3.7 Round-Trip Identity

The returned aggregates are **recalculated from the rounded hourly plan** so judges can re-verify from the response payload alone:

$$
\text{total\_grid\_kwh} = \sum_{h=0}^{23} g_h, \qquad
\text{total\_cost\_bdt} = \sum_{h=0}^{23} \tau_h \cdot g_h, \qquad
\text{peak\_grid\_kwh} = \max_{h} g_h
$$

with all values rounded to 4 decimal places to meet the official $\pm 0.01$ BDT / kWh tolerance.

---

## 4. Local Quickstart (Zero-Friction Reproducibility)

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

## 5. Deploy to Render (One-Click Blueprint)

The repo ships a [`render.yaml`](render.yaml) Blueprint that provisions the service in a single click.

### 5.1 One-time setup
1. Sign in to [dashboard.render.com](https://dashboard.render.com) with the GitHub account that owns `shaymimran45/gridwise-llm`.
2. Click **New** → **Blueprint**.
3. Select the **`shaymimran45/gridwise-llm`** repo (Render will read `render.yaml` automatically).
4. Click **Apply**. Render creates the `gridwise-llm-v1` web service and begins building.

### 5.2 Inject the Gemini API key
After the first build completes:
1. Open the new `gridwise-llm-v1` service in Render.
2. Go to **Environment** → **Add Environment Variable**.
3. Add:
   * **Key**: `GEMINI_API_KEY`
   * **Value**: paste your key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
4. Click **Save Changes** → Render auto-redeploys in ~30 seconds.

### 5.3 Verify the deployment
Once the service status shows **Live**:

```bash
# 1. Readiness check (judges can hit this directly)
curl https://gridwise-llm-v1.onrender.com/health
# Expected: {"status":"ok"}

# 2. Interactive dashboard
open https://gridwise-llm-v1.onrender.com/

# 3. End-to-end optimization call
curl -X POST https://gridwise-llm-v1.onrender.com/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "RENDER-CHECK",
    "operator_notes": ["Solar output drops by 50% from 1 PM to 3 PM"],
    "hours": [
      {"hour": h, "demand_kwh": 180, "solar_kwh": 80 if 6 <= h <= 18 else 0,
       "tariff_bdt_per_kwh": 7 + (h % 4)}
      for h in range(24)
    ],
    "battery": {
      "capacity_kwh": 500, "initial_energy_kwh": 200,
      "minimum_energy_kwh": 50,
      "max_charge_kwh_per_hour": 100,
      "max_discharge_kwh_per_hour": 100
    }
  }'
```

### 5.4 Free-tier caveats

| Concern | Behavior on Free Plan |
| :--- | :--- |
| **Cold start** | Service spins down after **15 minutes** of inactivity. First request after spin-down takes ~30-60 s. |
| **Mitigation** | Hit `/health` from an external uptime monitor (e.g. [UptimeRobot](https://uptimerobot.com), free) every 10 minutes to keep it warm during judging windows. |
| **RAM** | 512 MB. Single uvicorn worker (set in `start.sh`) — multi-worker would OOM. The async event loop + `asyncio.to_thread` already parallelizes requests within one worker. |
| **Bandwidth** | 100 GB/month — generous for an API + dashboard. |
| **Always-on alternative** | Upgrade to **Starter ($7/mo)** for instant cold-starts and no spin-down. |

### 5.5 What `render.yaml` configures

| Setting | Value | Why |
| :--- | :--- | :--- |
| `runtime` | `python` | Native Python build (faster than Docker on free plan). |
| `plan` | `free` | $0/month; sufficient for hackathon judging. |
| `region` | `oregon` | Lowest latency from Render's US edge. |
| `healthCheckPath` | `/health` | Render pings this every 30 s to confirm liveness. |
| `autoDeploy` | `true` | Every push to `master` triggers a redeploy. |
| `GEMINI_API_KEY` | `sync: false` | Stored as a Render secret (encrypted at rest), **never** in git. |

---

## 6. Docker Fallback Execution

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

## 6. Implementation Details & Technologies

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **Language Model** | Google Gemini (`gemini-2.5-flash-lite` / `gemini-3.6-flash`) | Low p95 latency (<2s), high structured JSON compliance, handles paraphrasing seamlessly. |
| **Deterministic Guardrails** | Custom Python Engine (`guardrails.py`) | Enforces 100% adherence to Section 08 rules (unique ascending hours 0-23, bounded numeric values, applies semantics). |
| **Optimizer / Solver** | HiGHS Linear Programming (`scipy.optimize.linprog`) | Solves 120-variable LP in <2ms with mathematical optimality, zero floating drift, and guaranteed energy balance. |
| **Web Framework** | FastAPI + Pydantic v2 | High throughput, asynchronous non-blocking I/O, strict automated schema validation. |

---

## 7. Security & Secret Handling
* **No hardcoded secrets**: API keys and tokens are loaded strictly via environment variables.
* **Safe logging**: Prompts, API responses, and logs never dump credential tokens or sensitive stack traces.
* **Safe failures**: Malformed inputs return HTTP 400 with structured validation errors, preventing 500 crashes.

---

## 8. Known Limitations & Edge Cases
* Time intervals in the Problem Statement are start-inclusive and end-exclusive (e.g. `1 PM to 3 PM` $\rightarrow$ `[13, 14]`). Minutes are rounded to nearest whole hours.
* Floating-point numbers are rounded to 4 decimal places in response JSON and meet the canonical $\pm 0.01$ kWh / BDT tolerance rule.

