import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
from llm_interpreter import interpret_operator_notes
from guardrails import validate_and_guardrail_directives
from optimizer import solve_energy_optimization


def _resolve_sample_file() -> Path:
    """Locate the public sample-cases JSON regardless of OS or working directory."""
    here = Path(__file__).parent
    candidates = [
        here / "Participant_Docs" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
        here.parent / "Participant_Docs" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
        Path.cwd() / "Participant_Docs" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Public sample cases JSON not found. Tried: "
        + " | ".join(str(p) for p in candidates)
    )


def load_cases() -> List[Dict[str, Any]]:
    sample_file = _resolve_sample_file()
    with open(sample_file, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def validate_directives(case: Dict[str, Any]) -> bool:
    """Return True iff every operator note is interpreted correctly."""
    notes = case["input"]["operator_notes"]
    cap = case["input"]["battery"]["capacity_kwh"]
    raw = interpret_operator_notes(notes, cap)
    val = validate_and_guardrail_directives(raw, notes, cap)
    for i, exp_dir in enumerate(case["expected_output"]["directive_interpretation"]):
        got = val[i]
        if not (
            exp_dir["applies"] == got.get("applies")
            and exp_dir["directive_type"] == got.get("directive_type")
            and exp_dir["structured_adjustment"] == got.get("structured_adjustment")
        ):
            return False
    return True


def validate_physics(case: Dict[str, Any], opt: Dict[str, Any]) -> Tuple[bool, str]:
    """Check energy balance, battery transitions, end-of-day neutrality."""
    inp = case["input"]
    init_energy = inp["battery"]["initial_energy_kwh"]
    hours_input = {h["hour"]: h for h in inp["hours"]}
    effective_solar = {h: hours_input[h]["solar_kwh"] for h in range(24)}
    for d in opt["_validated_directives"]:
        if d.get("applies") and d.get("directive_type") == "solar_reduction":
            factor = d["structured_adjustment"]["factor"]
            for h in d["structured_adjustment"]["hours"]:
                effective_solar[h] *= factor

    prev_energy = init_energy
    for entry in opt["hourly_plan"]:
        h = entry["hour"]
        dem = hours_input[h]["demand_kwh"]
        g = entry["grid_kwh"]
        s = entry["solar_used_kwh"]
        act = entry["battery_action"]
        b_kwh = entry["battery_kwh"]
        e_after = entry["battery_energy_after_kwh"]

        if s > effective_solar[h] + 0.01:
            return False, f"Hour {h}: solar_used ({s}) > effective_solar ({effective_solar[h]})"
        c_val = b_kwh if act == "charge" else 0.0
        d_val = b_kwh if act == "discharge" else 0.0
        if abs(g + s + d_val - (dem + c_val)) > 0.05:
            return False, f"Hour {h}: energy balance violation"
        expected_e = prev_energy + c_val - d_val
        if abs(e_after - expected_e) > 0.05:
            return False, f"Hour {h}: battery transition violation"
        prev_energy = e_after

    if abs(opt["hourly_plan"][23]["battery_energy_after_kwh"] - init_energy) > 0.05:
        return False, "End-of-day neutrality violated"
    return True, ""


def run_full_validation():
    cases = load_cases()
    print("=" * 80)
    print("  BUP CSE FEST 2026 - GRIDWISE LLM PUBLIC SAMPLES VERIFICATION")
    print("=" * 80)

    total_cases = len(cases)
    passed_cases = 0
    total_notes = sum(len(c["input"]["operator_notes"]) for c in cases)
    passed_notes = 0

    for c in cases:
        c_id = c["id"]
        inp = c["input"]
        exp = c["expected_output"]
        cap = inp["battery"]["capacity_kwh"]

        raw = interpret_operator_notes(inp["operator_notes"], cap)
        val = validate_and_guardrail_directives(raw, inp["operator_notes"], cap)
        passed_notes += sum(
            1 for i, ed in enumerate(exp["directive_interpretation"])
            if val[i].get("applies") == ed["applies"]
            and val[i].get("directive_type") == ed["directive_type"]
            and val[i].get("structured_adjustment") == ed["structured_adjustment"]
        )

        opt = solve_energy_optimization(c_id, inp["hours"], inp["battery"], val)
        opt["_validated_directives"] = val

        notes_ok = validate_directives(c)
        physics_ok, physics_err = validate_physics(c, opt)
        cost_diff = abs(exp["total_cost_bdt"] - opt["total_cost_bdt"])
        cost_ok = cost_diff <= 0.05

        all_ok = notes_ok and physics_ok and cost_ok
        if all_ok:
            passed_cases += 1
            print(f"[{c_id}] PASS | Notes: OK | Physics: OK | Cost: {opt['total_cost_bdt']:.2f} BDT (Diff: {cost_diff:.4f})")
        else:
            print(
                f"[{c_id}] FAIL | Notes: {'OK' if notes_ok else 'FAIL'} | "
                f"Physics: {'OK' if physics_ok else physics_err} | Cost Diff: {cost_diff:.4f}"
            )

    print("=" * 80)
    print(f"OVERALL SUMMARY: Cases: {passed_cases}/{total_cases} passed | Notes: {passed_notes}/{total_notes} passed")
    print("=" * 80)


# Pytest-compatible fixtures & tests
def pytest_cases():
    """Lazy import helper to keep `python test_samples.py` working without pytest."""
    import pytest
    return pytest.param


try:
    import pytest  # type: ignore
    _CASES = load_cases()
    _CASE_IDS = [c["id"] for c in _CASES]

    @pytest.fixture(scope="module")
    def cases():
        return load_cases()

    @pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
    def test_directive_interpretation(case):
        assert validate_directives(case), f"Directive mismatch in {case['id']}"

    @pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
    def test_physics(case):
        inp = case["input"]
        raw = interpret_operator_notes(inp["operator_notes"], inp["battery"]["capacity_kwh"])
        val = validate_and_guardrail_directives(raw, inp["operator_notes"], inp["battery"]["capacity_kwh"])
        opt = solve_energy_optimization(case["id"], inp["hours"], inp["battery"], val)
        opt["_validated_directives"] = val
        ok, err = validate_physics(case, opt)
        assert ok, f"Physics violation in {case['id']}: {err}"

    @pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
    def test_cost(case):
        inp = case["input"]
        raw = interpret_operator_notes(inp["operator_notes"], inp["battery"]["capacity_kwh"])
        val = validate_and_guardrail_directives(raw, inp["operator_notes"], inp["battery"]["capacity_kwh"])
        opt = solve_energy_optimization(case["id"], inp["hours"], inp["battery"], val)
        diff = abs(opt["total_cost_bdt"] - case["expected_output"]["total_cost_bdt"])
        assert diff <= 0.05, f"Cost diff {diff:.4f} in {case['id']}"

except ImportError:
    pass  # pytest not installed; the __main__ branch still works


if __name__ == "__main__":
    run_full_validation()


