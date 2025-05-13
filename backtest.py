import pandas as pd
import numpy as np
import itertools
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# === Parameters ===
ORDER_SIZE = 5000
STEP_SIZE = 100
PARAM_GRID = {
    'lambda_under': [0.01, 0.05, 0.1],
    'lambda_over': [0.01, 0.05, 0.1],
    'theta_queue': [0.0001, 0.001, 0.01]
}

# === Load L1 Data ===
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
            'fee': 0.003,
            'rebate': 0.001
        })
    return list(snapshots.items())

# === Cost Function ===
def compute_cost(split, venues, order_size, lam_over, lam_under, theta):
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
    penalty = lam_under * underfill + lam_over * overfill + theta * (underfill + overfill)
    return cash_spent + penalty

# === Allocator ===
def allocate(order_size, venues, lam_over, lam_under, theta):
    splits = [[]]
    for v in venues:
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_add = min(order_size - used, v['ask_size'])
            for q in range(0, int(max_add) + 1, STEP_SIZE):
                new_splits.append(alloc + [q])
        splits = new_splits
    best_cost, best_split = float('inf'), []
    for split in splits:
        if sum(split) != order_size:
            continue
        cost = compute_cost(split, venues, order_size, lam_over, lam_under, theta)
        if cost < best_cost:
            best_cost, best_split = cost, split
    return best_split, best_cost

# === Execute Split ===
def execute_split(split, venues):
    total_cost, executed = 0, 0
    for i, qty in enumerate(split):
        venue = venues[i]
        fill = min(qty, venue['ask_size'])
        total_cost += fill * (venue['ask'] + venue['fee'])
        executed += fill
    return executed, total_cost

# === Allocator Backtest ===
def run_allocator_backtest(snapshots, lam_over, lam_under, theta):
    remaining = ORDER_SIZE
    total_cost, total_exec = 0, 0
    for ts, venues in snapshots:
        if remaining <= 0:
            break
        alloc, _ = allocate(remaining, venues, lam_over, lam_under, theta)
        filled, cost = execute_split(alloc, venues)
        total_cost += cost
        total_exec += filled
        remaining -= filled
    avg_price = total_cost / total_exec if total_exec else None
    return total_cost, avg_price

# === Parameter Search ===
def search_parameters(snapshots):
    best_params = None
    best_total_cost = float('inf')
    best_avg_price = None
    for lam_u, lam_o, theta in itertools.product(
        PARAM_GRID['lambda_under'],
        PARAM_GRID['lambda_over'],
        PARAM_GRID['theta_queue']
    ):
        cost, avg_price = run_allocator_backtest(snapshots, lam_o, lam_u, theta)
        if cost < best_total_cost:
            best_total_cost = cost
            best_avg_price = avg_price
            best_params = {'lambda_under': lam_u, 'lambda_over': lam_o, 'theta_queue': theta}
    return best_params, best_total_cost, best_avg_price

# === Baseline Strategies ===
def best_ask_strategy(snapshots):
    remaining, cost, cumulative = ORDER_SIZE, 0, []
    for ts, venues in snapshots:
        if remaining <= 0:
            break
        best = min(venues, key=lambda v: v['ask'])
        qty = min(remaining, best['ask_size'])
        cost += qty * (best['ask'] + best['fee'])
        cumulative.append(cost)
        remaining -= qty
    return cumulative

def twap_strategy(snapshots):
    n_slices = 9
    per_slice = ORDER_SIZE // n_slices
    chunk_len = len(snapshots) // n_slices
    ts_buckets = [snapshots[i*chunk_len:(i+1)*chunk_len] for i in range(n_slices)]
    remaining, cost, cumulative = ORDER_SIZE, 0, []
    for bucket in ts_buckets:
        slice_qty = min(per_slice, remaining)
        for ts, venues in bucket:
            if remaining <= 0 or slice_qty <= 0:
                break
            best = min(venues, key=lambda v: v['ask'])
            qty = min(slice_qty, best['ask_size'])
            cost += qty * (best['ask'] + best['fee'])
            cumulative.append(cost)
            remaining -= qty
            slice_qty -= qty
    return cumulative

def vwap_strategy(snapshots):
    remaining, cost, cumulative = ORDER_SIZE, 0, []
    for ts, venues in snapshots:
        if remaining <= 0:
            break
        total_liq = sum(v['ask_size'] for v in venues)
        if total_liq == 0:
            continue
        for v in venues:
            weight = v['ask_size'] / total_liq
            alloc = min(remaining, int(weight * ORDER_SIZE))
            fill = min(alloc, v['ask_size'])
            cost += fill * (v['ask'] + v['fee'])
            cumulative.append(cost)
            remaining -= fill
            if remaining <= 0:
                break
    return cumulative

# === Plot Strategy ===
def plot_strategy(series, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(series)
    plt.title(f'Cumulative Cost: {title}')
    plt.xlabel('Snapshot')
    plt.ylabel('Cumulative Cost ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# === Main ===
def main():
    snapshots = load_data('l1_day.csv')
    best_params, opt_cost, opt_avg_price = search_parameters(snapshots)

    best_ask_series = best_ask_strategy(snapshots)
    twap_series = twap_strategy(snapshots)
    vwap_series = vwap_strategy(snapshots)

    plot_strategy(best_ask_series, "Best Ask", "best_ask_plot.png")
    plot_strategy(twap_series, "TWAP", "twap_plot.png")
    plot_strategy(vwap_series, "VWAP", "vwap_plot.png")

    result = {
        'best_params': best_params,
        'optimal_total_cost': opt_cost,
        'optimal_avg_fill_price': opt_avg_price
    }
    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Plots and results saved.")

if __name__ == '__main__':
    main()
