import logging
import asyncio
import time
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, model_validator

from config import PORT, HOST
from guardrails import validate_and_guardrail_directives
from optimizer import solve_energy_optimization

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GridWise")


# In-memory caches for static dashboard assets (loaded at startup)
_INDEX_HTML_CACHE: Optional[str] = None
_SAMPLES_CACHE: Optional[Dict[str, Any]] = None


def _load_index_html() -> str:
    global _INDEX_HTML_CACHE
    if _INDEX_HTML_CACHE is None:
        html_file = BASE_DIR / "static" / "index.html"
        if html_file.exists():
            _INDEX_HTML_CACHE = html_file.read_text(encoding="utf-8")
        else:
            _INDEX_HTML_CACHE = "<h1>GridWise LLM Service Running</h1>"
    return _INDEX_HTML_CACHE


def _load_samples() -> Dict[str, Any]:
    global _SAMPLES_CACHE
    if _SAMPLES_CACHE is None:
        candidates = [
            BASE_DIR / "data" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
            BASE_DIR / "Participant_Docs" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
        ]
        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        _SAMPLES_CACHE = json.load(f)
                    return _SAMPLES_CACHE
                except Exception:
                    continue
        _SAMPLES_CACHE = {"cases": []}
    return _SAMPLES_CACHE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load static assets and warm the persistent LLM client at startup."""
    _load_index_html()
    _load_samples()
    # Warm persistent httpx client (so first request doesn't pay TLS handshake cost)
    try:
        from llm_interpreter import get_async_client
        get_async_client()
    except Exception:
        pass
    logger.info("GridWise LLM startup complete: index.html + samples cached, async client warmed.")
    yield
    # Graceful shutdown: close persistent httpx client.
    try:
        from llm_interpreter import close_async_client
        await close_async_client()
    except Exception:
        pass


app = FastAPI(
    title="GridWise LLM - Smart Campus Energy Optimization",
    description="Automated LLM-assisted energy scheduling API for BUP CSE Fest 2026 Hackathon",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a short correlation ID and process-time header to every request."""
    import uuid
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["x-request-id"] = rid
    response.headers["x-process-time-ms"] = f"{elapsed_ms:.1f}"
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    from fastapi.encoders import jsonable_encoder
    rid = getattr(request.state, "request_id", "-")
    logger.warning(f"[{rid}] Request validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Malformed JSON or structurally invalid request",
            "errors": jsonable_encoder(exc.errors()),
            "request_id": rid,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "-")
    logger.error(f"[{rid}] Internal server error on {request.method} {request.url.path}: {exc}", exc_info=False)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal error occurred during processing.", "request_id": rid}
    )


# Request Schemas
class HourInput(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class BatteryInput(BaseModel):
    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _physical_consistency(self):
        """Bug fix #4: ensure battery specs are physically consistent before reaching LP solver.
        Prevents 500s on infeasible inputs and gives judges a clear 400 message."""
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError(
                f"battery.initial_energy_kwh ({self.initial_energy_kwh}) must be <= capacity_kwh ({self.capacity_kwh})"
            )
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError(
                f"battery.minimum_energy_kwh ({self.minimum_energy_kwh}) must be <= capacity_kwh ({self.capacity_kwh})"
            )
        if self.minimum_energy_kwh > self.initial_energy_kwh:
            # Feasible but limits solver to never charge then discharge before EOD —
            # we still accept it but log a warning.
            logger.warning(
                "battery.minimum_energy_kwh (%.2f) > initial_energy_kwh (%.2f) — schedule will charge before reserve applies",
                self.minimum_energy_kwh,
                self.initial_energy_kwh,
            )
        return self


class OptimizeEnergyRequest(BaseModel):
    scenario_id: str = Field(..., min_length=1, max_length=128)
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourInput] = Field(..., min_length=24, max_length=24)
    battery: BatteryInput

    @model_validator(mode="after")
    def _no_blank_notes(self):
        for i, n in enumerate(self.operator_notes):
            if not isinstance(n, str) or not n.strip():
                raise ValueError(f"operator_notes[{i}] must be a non-empty string")
            if len(n) > 2000:
                raise ValueError(f"operator_notes[{i}] exceeds 2000-character limit")
        return self


# Response Schemas
class HealthResponse(BaseModel):
    status: str = "ok"


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeEnergyResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


BASE_DIR = Path(__file__).parent


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Interactive Web Dashboard for operators and judges."""
    return _load_index_html()


@app.get("/api/samples")
async def get_sample_cases():
    """Returns public sample cases to populate the interactive dashboard."""
    data = _load_samples()
    return {"cases": data.get("cases", [])}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Readiness endpoint for the judging harness. Must return HTTP 200 with status: ok."""
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse)
async def optimize_energy(payload: OptimizeEnergyRequest):
    """
    Main LLM interpretation + 24-hour energy optimization endpoint.
    - LLM call is async + persistent-connection-pooled (low p95 latency).
    - Solver runs on a worker thread so the asyncio loop stays non-blocking.
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Processing optimization request for scenario: {payload.scenario_id}")

    # 1. Validate hours length & uniqueness
    hours_dict = [h.model_dump() for h in payload.hours]
    unique_hours = set(h["hour"] for h in hours_dict)
    if len(unique_hours) != 24 or set(range(24)) != unique_hours:
        raise HTTPException(
            status_code=400,
            detail="hours must contain exactly 24 unique hours from 0 to 23.",
        )

    battery_dict = payload.battery.model_dump()
    operator_notes = payload.operator_notes

    # 2. LLM interpretation (async, non-blocking; safe fallback on failure)
    try:
        from llm_interpreter import interpret_operator_notes_async
        raw_directives = await interpret_operator_notes_async(
            operator_notes,
            battery_dict["capacity_kwh"],
        )
    except Exception as e:
        logger.error(f"[{request_id}] LLM interpretation error: {e}")
        from llm_interpreter import fallback_interpret_note
        raw_directives = [
            fallback_interpret_note(n, i, battery_dict["capacity_kwh"])
            for i, n in enumerate(operator_notes)
        ]

    # 3. Deterministic Guardrails
    validated_directives = validate_and_guardrail_directives(
        raw_directives,
        operator_notes,
        battery_dict["capacity_kwh"],
    )

    # 4. Mathematical Optimization — offload to a worker thread so the event loop stays free
    import asyncio as _aio
    try:
        opt_result = await _aio.to_thread(
            solve_energy_optimization,
            payload.scenario_id,
            hours_dict,
            battery_dict,
            validated_directives,
        )
    except RuntimeError as e:
        # Infeasible LP — return a 422 with structured detail so judges can debug.
        logger.warning(f"[{request_id}] Optimization infeasible: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Optimization infeasible",
                "reason": str(e),
                "scenario_id": payload.scenario_id,
                "hint": "Directives may be mutually contradictory (e.g. max_grid=0 + high demand + no_discharge).",
            },
        )
    except Exception as e:
        logger.error(f"[{request_id}] Optimization solving error: {e}", exc_info=False)
        raise HTTPException(status_code=500, detail="Optimization solver encountered an error.")

    # 5. Assemble response matching canonical contract
    logger.info(
        f"[{request_id}] OK | scenario={payload.scenario_id} | cost={opt_result['total_cost_bdt']:.2f} BDT"
    )
    return {
        "scenario_id": payload.scenario_id,
        "directive_interpretation": validated_directives,
        "hourly_plan": opt_result["hourly_plan"],
        "total_grid_kwh": opt_result["total_grid_kwh"],
        "total_cost_bdt": opt_result["total_cost_bdt"],
        "peak_grid_kwh": opt_result["peak_grid_kwh"],
        "plan_summary": opt_result["plan_summary"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

