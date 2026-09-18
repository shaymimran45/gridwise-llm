from typing import List, Dict, Any
from llm_interpreter import fallback_interpret_note

ALLOWED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op"
}


def validate_and_guardrail_directives(
    raw_directives: List[Dict[str, Any]],
    operator_notes: List[str],
    battery_capacity_kwh: float
) -> List[Dict[str, Any]]:
    """
    Deterministic Guardrails according to Section 08 of the Problem Statement.
    Enforces exact schema, unique ascending hours 0-23, bounded numeric values,
    correct applies semantics, and safe controlled fallback.
    """
    cleaned: List[Dict[str, Any]] = []
    
    # Ensure we process each note index 0..N-1
    by_index = {item.get("note_index"): item for item in raw_directives if isinstance(item, dict) and "note_index" in item}

    for idx, note_text in enumerate(operator_notes):
        item = by_index.get(idx)
        
        # If missing or malformed, fallback to deterministic parser
        if not item or not isinstance(item, dict):
            cleaned.append(fallback_interpret_note(note_text, idx, battery_capacity_kwh))
            continue

        dtype = item.get("directive_type")
        if dtype not in ALLOWED_DIRECTIVES:
            cleaned.append(fallback_interpret_note(note_text, idx, battery_capacity_kwh))
            continue

        applies = item.get("applies", False)
        adj = item.get("structured_adjustment")
        explanation = item.get("explanation", "") or "Interpreted directive."

        # Case 1: no_op
        if dtype == "no_op":
            cleaned.append({
                "note_index": idx,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": str(explanation)
            })
            continue

        # Case 2: active directives (applies must be True)
        if not isinstance(adj, dict) or not applies:
            # If marked active but adj is invalid or applies is false, re-parse with fallback
            fb = fallback_interpret_note(note_text, idx, battery_capacity_kwh)
            cleaned.append(fb)
            continue

        # Validate hours
        raw_hours = adj.get("hours", [])
        if not isinstance(raw_hours, list):
            cleaned.append(fallback_interpret_note(note_text, idx, battery_capacity_kwh))
            continue

        valid_hours = []
        for h in raw_hours:
            try:
                h_int = int(h)
                if 0 <= h_int <= 23:
                    valid_hours.append(h_int)
            except (ValueError, TypeError):
                continue
        
        valid_hours = sorted(list(set(valid_hours)))
        if not valid_hours:
            cleaned.append(fallback_interpret_note(note_text, idx, battery_capacity_kwh))
            continue

        # Validate directive-specific fields
        if dtype == "solar_reduction":
            factor = adj.get("factor")
            try:
                factor_f = float(factor)
                factor_f = max(0.0, min(1.0, factor_f))
            except (ValueError, TypeError):
                cleaned.append(fallback_interpret_note(note_text, idx, battery_capacity_kwh))
                continue

            cleaned.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {
                    "hours": valid_hours,
                    "factor": factor_f
                },
                "explanation": str(explanation)
            })

        elif dtype == "minimum_battery_reserve":
            reserve = adj.get("minimum_energy_kwh")
            try:
                reserve_f = float(reserve)
                reserve_f = max(0.0, min(battery_capacity_kwh, reserve_f))
            except (ValueError, TypeError):
                cleaned.append(fallback_interpret_note(note_text, idx, battery_capacity_kwh))
                continue

            cleaned.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "minimum_battery_reserve",
                "structured_adjustment": {
                    "hours": valid_hours,
                    "minimum_energy_kwh": reserve_f
                },
                "explanation": str(explanation)
            })

        elif dtype == "max_grid_window":
            grid_cap = adj.get("max_grid_kwh")
            try:
                grid_cap_f = float(grid_cap)
                grid_cap_f = max(0.0, grid_cap_f)
            except (ValueError, TypeError):
                cleaned.append(fallback_interpret_note(note_text, idx, battery_capacity_kwh))
                continue

            cleaned.append({
                "note_index": idx,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {
                    "hours": valid_hours,
                    "max_grid_kwh": grid_cap_f
                },
                "explanation": str(explanation)
            })

        elif dtype in ("no_charge_window", "no_discharge_window"):
            cleaned.append({
                "note_index": idx,
                "applies": True,
                "directive_type": dtype,
                "structured_adjustment": {
                    "hours": valid_hours
                },
                "explanation": str(explanation)
            })

    return cleaned

