
# README.md — Smart Order Routing Backtest

## Approach Summary

This project implements a backtest of a Smart Order Router (SOR) that allocates a 5,000-share parent buy order across multiple venues. The routing logic follows a static optimization model inspired by Cont & Kukanov (2014). At each quote update (L1 snapshot), the system determines the optimal split of the remaining order based on venue liquidity, price, fees, and penalties for execution risk.

For every new market snapshot:

- We observe venue prices and available sizes.
- We compute all valid allocations (`splits`) of remaining shares across venues.
- For each split, we calculate the expected total cost:
  - Execution cost (price + fee)
  - Rebates (if some shares remain unfilled)
  - Penalties for underfill, overfill, and queue risk
- The split with the lowest cost is executed.

This is compared against three baseline strategies:

- Best Ask: Fill greedily from the cheapest venue.
- TWAP: Execute evenly over time.
- VWAP: Allocate proportionally to visible liquidity.

## Parameter Ranges (Grid Search)

We perform a grid search over combinations of the following penalty parameters:

| Parameter      | Values Tested         | Purpose                                        |
| -------------- | --------------------- | ---------------------------------------------- |
| `lambda_under` | [0.01, 0.05, 0.1]     | Penalty for failing to fully execute the order |
| `lambda_over`  | [0.01, 0.05, 0.1]     | Penalty for buying more than target size       |
| `theta_queue`  | [0.0001, 0.001, 0.01] | Penalty for execution uncertainty (queue risk) |

The best-performing configuration is selected based on total cost over the simulation horizon.

## Improving Fill Realism — One Idea

Incorporate a model of queue position and slippage to reflect probabilistic fills.

In the current simulation, we assume that we can fill up to the visible `ask_size` at each venue. However, in reality, order execution depends on your position in the queue and other order flow activity.

Proposed enhancement:

- Track prior ask depth over time.
- Assign lower fill probability to passive orders deeper in the queue.
- Introduce slippage for aggressive orders due to short-term price movements.

This would make the simulator more realistic by accounting for latency, competition, and adverse selection, and allow dynamic adjustment of the `theta_queue` penalty.

## To Run

```bash
python3 backtest.py
```

Outputs:

- JSON summary of costs and optimal parameters
- `results.png`: cumulative cost plot of all strategies
