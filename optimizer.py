import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Any, Tuple


def solve_energy_optimization(
    scenario_id: str,
    hours_data: List[Dict[str, Any]],
    battery_data: Dict[str, Any],
    directives: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Solves the 24-hour smart campus energy schedule using exact Linear Programming (HiGHS).
    Guarantees global mathematical optimality, energy balance, and constraint compliance.
    """
    T = 24
    
    # Sort hours by hour 0..23
    sorted_hours = sorted(hours_data, key=lambda x: x["hour"])
    demands = np.array([float(h["demand_kwh"]) for h in sorted_hours])
    solars = np.array([float(h["solar_kwh"]) for h in sorted_hours])
    tariffs = np.array([float(h["tariff_bdt_per_kwh"]) for h in sorted_hours])

    # Battery specifications
    capacity = float(battery_data["capacity_kwh"])
    initial_energy = float(battery_data["initial_energy_kwh"])
    base_min_energy = float(battery_data["minimum_energy_kwh"])
    max_charge_rate = float(battery_data["max_charge_kwh_per_hour"])
    max_discharge_rate = float(battery_data["max_discharge_kwh_per_hour"])

    # Effective limits initialized
    effective_solar = np.copy(solars)
    min_reserves = np.full(T, base_min_energy)
    max_charges = np.full(T, max_charge_rate)
    max_discharges = np.full(T, max_discharge_rate)
    max_grids = np.full(T, np.inf)

    # Apply directives
    for d in directives:
        if not d.get("applies"):
            continue
        dtype = d.get("directive_type")
        adj = d.get("structured_adjustment") or {}
        d_hours = adj.get("hours", [])

        if dtype == "solar_reduction":
            factor = float(adj.get("factor", 1.0))
            for h in d_hours:
                if 0 <= h < T:
                    effective_solar[h] = effective_solar[h] * factor

        elif dtype == "minimum_battery_reserve":
            req_min = float(adj.get("minimum_energy_kwh", base_min_energy))
            for h in d_hours:
                if 0 <= h < T:
                    min_reserves[h] = max(min_reserves[h], req_min)

        elif dtype == "no_charge_window":
            for h in d_hours:
                if 0 <= h < T:
                    max_charges[h] = 0.0

        elif dtype == "no_discharge_window":
            for h in d_hours:
                if 0 <= h < T:
                    max_discharges[h] = 0.0

        elif dtype == "max_grid_window":
            grid_cap = float(adj.get("max_grid_kwh", np.inf))
            for h in d_hours:
                if 0 <= h < T:
                    max_grids[h] = min(max_grids[h], grid_cap)

    # Variable Index mapping:
    # 0..23: grid_kwh (g)
    # 24..47: solar_used_kwh (s)
    # 48..71: battery_charge (c)
    # 72..95: battery_discharge (d)
    # 96..119: battery_energy (E)
    n_vars = 5 * T
    g_idx = lambda h: h
    s_idx = lambda h: T + h
    c_idx = lambda h: 2 * T + h
    d_idx = lambda h: 3 * T + h
    e_idx = lambda h: 4 * T + h

    # Cost vector
    c = np.zeros(n_vars)
    # Grid cost
    for h in range(T):
        c[g_idx(h)] = tariffs[h]
        # Tiny tie-breaker (1e-7) on battery charge/discharge to prevent simultaneous or redundant cycling
        c[c_idx(h)] = 1e-7
        c[d_idx(h)] = 1e-7

    # Equality constraints: A_eq @ x = b_eq
    eq_rows = []
    b_eq = []

    # 1. Energy balance: g_h + s_h + d_h - c_h = demand_h
    for h in range(T):
        row = np.zeros(n_vars)
        row[g_idx(h)] = 1.0
        row[s_idx(h)] = 1.0
        row[d_idx(h)] = 1.0
        row[c_idx(h)] = -1.0
        eq_rows.append(row)
        b_eq.append(demands[h])

    # 2. Battery state transition:
    # h = 0: E_0 - c_0 + d_0 = initial_energy
    row0 = np.zeros(n_vars)
    row0[e_idx(0)] = 1.0
    row0[c_idx(0)] = -1.0
    row0[d_idx(0)] = 1.0
    eq_rows.append(row0)
    b_eq.append(initial_energy)

    # h = 1..23: E_h - E_{h-1} - c_h + d_h = 0
    for h in range(1, T):
        row = np.zeros(n_vars)
        row[e_idx(h)] = 1.0
        row[e_idx(h - 1)] = -1.0
        row[c_idx(h)] = -1.0
        row[d_idx(h)] = 1.0
        eq_rows.append(row)
        b_eq.append(0.0)

    # 3. End-of-day neutrality: E_23 = initial_energy
    row_eod = np.zeros(n_vars)
    row_eod[e_idx(23)] = 1.0
    eq_rows.append(row_eod)
    b_eq.append(initial_energy)

    A_eq = np.array(eq_rows)
    b_eq = np.array(b_eq)

    # Bounds on variables
    bounds = []
    for h in range(T):
        upper_g = max_grids[h] if np.isfinite(max_grids[h]) else None
        bounds.append((0.0, upper_g))
    for h in range(T):
        bounds.append((0.0, effective_solar[h]))
    for h in range(T):
        bounds.append((0.0, max_charges[h]))
    for h in range(T):
        bounds.append((0.0, max_discharges[h]))
    for h in range(T):
        bounds.append((min_reserves[h], capacity))

    # Solve with HiGHS solver
    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True}
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed for scenario {scenario_id}: {res.message}")

    x = res.x
    hourly_plan = []
    total_grid_kwh = 0.0
    total_cost_bdt = 0.0
    peak_grid_kwh = 0.0

    for h in range(T):
        g_val = max(0.0, x[g_idx(h)])
        s_val = max(0.0, x[s_idx(h)])
        c_val = max(0.0, x[c_idx(h)])
        d_val = max(0.0, x[d_idx(h)])
        e_val = max(0.0, x[e_idx(h)])

        # Determine battery action
        if c_val > 1e-4:
            action = "charge"
            b_kwh = c_val
        elif d_val > 1e-4:
            action = "discharge"
            b_kwh = d_val
        else:
            action = "idle"
            b_kwh = 0.0

        # Rounding for exactness
        g_val_r = round(g_val, 4)
        s_val_r = round(s_val, 4)
        b_kwh_r = round(b_kwh, 4)
        e_val_r = round(e_val, 4)

        # Totals recalculation
        total_grid_kwh += g_val_r
        total_cost_bdt += g_val_r * tariffs[h]
        if g_val_r > peak_grid_kwh:
            peak_grid_kwh = g_val_r

        hourly_plan.append({
            "hour": h,
            "grid_kwh": g_val_r,
            "solar_used_kwh": s_val_r,
            "battery_action": action,
            "battery_kwh": b_kwh_r,
            "battery_energy_after_kwh": e_val_r
        })

    # Final recalculations
    total_grid_kwh = round(total_grid_kwh, 4)
    total_cost_bdt = round(total_cost_bdt, 4)
    peak_grid_kwh = round(peak_grid_kwh, 4)

    plan_summary = (
        f"Optimized 24-hour schedule for {scenario_id}: total grid import {total_grid_kwh:.2f} kWh, "
        f"total cost {total_cost_bdt:.2f} BDT, peak grid demand {peak_grid_kwh:.2f} kWh. "
        f"All battery reserve, outage windows, and solar directives satisfied with end-of-day neutrality."
    )

    return {
        "hourly_plan": hourly_plan,
        "total_grid_kwh": total_grid_kwh,
        "total_cost_bdt": total_cost_bdt,
        "peak_grid_kwh": peak_grid_kwh,
        "plan_summary": plan_summary
    }

