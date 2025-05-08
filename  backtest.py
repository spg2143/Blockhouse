import pandas as pd
import numpy as np
import itertools
import json
from collections import defaultdict

# ====== Parameters ======
ORDER_SIZE = 5000
STEP_SIZE = 100
PARAM_GRID = {
    'lambda_under': [0.01, 0.05, 0.1],
    'lambda_over': [0.01, 0.05, 0.1],
    'theta_queue': [0.0001, 0.001, 0.01]
}

# ====== Data Loading ======
def load_data(path):
    df = pd.read_csv(path)
    df = df.sort_values('ts_event')
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'])
    snapshots = defaultdict(list)
    for _, row in df.iterrows():
        snapshots[row['ts_event']].append({
            'venue': row['publisher_id'],
            'ask': row['ask_px_00'],
            'ask_size': row['ask_sz_00'],
            'fee': 0.003,  # stub
            'rebate': 0.001  # stub
        })
    return list(snapshots.items())

# ====== Cost Computation ======
def compute_cost(split, venues, order_size, lam_o, lam_u, theta):
    executed, cash_spent = 0, 0
    for i, shares in enumerate(split):
        venue = venues[i]
        exe = min(shares, venue['ask_size'])
        executed += exe
        cash_spent += exe * (venue['ask'] + venue['fee'])
        rebate = max(shares - exe, 0) * venue['rebate']
        cash_spent -= rebate
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    return cash_spent + lam_u * underfill + lam_o * overfill + theta * (underfill + overfill)

# ====== Allocator ======
def allocate(order_size, venues, lam_o, lam_u, theta):
    splits = [[]]
    for v in venues:
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, v['ask_size'])
            for q in range(0, int(max_v) + 1, STEP_SIZE):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost, best_split = float('inf'), []
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lam_o, lam_u, theta)
        if cost < best_cost:
            best_cost, best_split = cost, alloc
    return best_split, best_cost

# ====== Execution Simulation ======
def execute_split(split, venues):
    fills = []
    total_cost, executed = 0, 0
    for i, qty in enumerate(split):
        venue = venues[i]
        fill_qty = min(qty, venue['ask_size'])
        cost = fill_qty * (venue['ask'] + venue['fee'])
        total_cost += cost
        fills.append(fill_qty)
        executed += fill_qty
    return executed, total_cost

# ====== Baseline Strategies ======
def best_ask_strategy(snapshots):
    remaining = ORDER_SIZE
    cost = 0
    for ts, venues in snapshots:
        best = min(venues, key=lambda v: v['ask'])
        qty = min(remaining, best['ask_size'])
        cost += qty * (best['ask'] + best['fee'])
        remaining -= qty
        if remaining <= 0:
            break
    return cost, ORDER_SIZE - remaining

# Placeholders (TWAP, VWAP)
def twap_strategy(snapshots):
    return best_ask_strategy(snapshots)  # placeholder

def vwap_strategy(snapshots):
    return best_ask_strategy(snapshots)  # placeholder

# ====== Parameter Search ======
def search_parameters(snapshots):
    best_params = None
    best_total_cost = float('inf')
    best_avg_price = None
    for lam_u, lam_o, theta in itertools.product(PARAM_GRID['lambda_under'],
                                                 PARAM_GRID['lambda_over'],
                                                 PARAM_GRID['theta_queue']):
        remaining = ORDER_SIZE
        total_cost, total_exec = 0, 0
        for ts, venues in snapshots:
            if remaining <= 0:
                break
            alloc, _ = allocate(remaining, venues, lam_o, lam_u, theta)
            filled, cost = execute_split(alloc, venues)
            remaining -= filled
            total_cost += cost
            total_exec += filled

        if total_exec == 0:
            continue
        avg_price = total_cost / total_exec
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_params = {'lambda_under': lam_u, 'lambda_over': lam_o, 'theta_queue': theta}
            best_avg_price = avg_price
    return best_params, best_total_cost, best_avg_price

# ====== Main ======
def main():
    snapshots = load_data('l1_day.csv')
    best_params, opt_cost, opt_avg_price = search_parameters(snapshots)
    best_cost, best_filled = best_ask_strategy(snapshots)
    twap_cost, _ = twap_strategy(snapshots)
    vwap_cost, _ = vwap_strategy(snapshots)

    def bps(save, ref):
        return 10000 * (ref - save) / ref if ref else 0

    result = {
        'best_params': best_params,
        'optimal_total_cost': opt_cost,
        'optimal_avg_fill_price': opt_avg_price,
        'baseline_best_ask': {'cost': best_cost},
        'baseline_twap': {'cost': twap_cost},
        'baseline_vwap': {'cost': vwap_cost},
        'savings_vs_best_ask_bps': bps(opt_cost, best_cost),
        'savings_vs_twap_bps': bps(opt_cost, twap_cost),
        'savings_vs_vwap_bps': bps(opt_cost, vwap_cost)
    }
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
