# Blockhouse

\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{listings}
\geometry{margin=1in}

\title{Smart Order Routing Backtest}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Approach Summary}

This project implements a backtest of a Smart Order Router (SOR) that allocates a 5,000-share parent buy order across multiple venues. The routing logic follows a static optimization model inspired by Cont \& Kukanov (2014). At each quote update (L1 snapshot), the system determines the optimal split of the remaining order based on venue liquidity, price, fees, and penalties for execution risk.

For every new market snapshot:
\begin{itemize}
  \item We observe venue prices and available sizes.
  \item We compute all valid allocations (\texttt{splits}) of remaining shares across venues.
  \item For each split, we calculate the expected total cost:
    \begin{itemize}
      \item Execution cost (price + fee)
      \item Rebates (if some shares remain unfilled)
      \item Penalties for underfill, overfill, and queue risk
    \end{itemize}
  \item The split with the lowest cost is executed.
\end{itemize}

This is compared against three baseline strategies:
\begin{itemize}
  \item \textbf{Best Ask}: Fill greedily from the cheapest venue.
  \item \textbf{TWAP}: Execute evenly over time.
  \item \textbf{VWAP}: Allocate proportionally to visible liquidity.
\end{itemize}

\section*{Parameter Ranges (Grid Search)}

We perform a grid search over combinations of the following penalty parameters:

\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Parameter} & \textbf{Values Tested} & \textbf{Purpose} \\
\midrule
\texttt{lambda\_under} & [0.01, 0.05, 0.1] & Penalty for failing to fully execute the order \\
\texttt{lambda\_over} & [0.01, 0.05, 0.1] & Penalty for buying more than target size \\
\texttt{theta\_queue} & [0.0001, 0.001, 0.01] & Penalty for execution uncertainty (queue risk) \\
\bottomrule
\end{tabular}

\vspace{1em}
The best-performing configuration is selected based on total cost over the simulation horizon.

\section*{Improving Fill Realism — One Idea}

\textbf{Incorporate a model of queue position and slippage to reflect probabilistic fills.}

In the current simulation, we assume that we can fill up to the visible \texttt{ask\_size} at each venue. However, in reality, order execution depends on your position in the queue and other order flow activity.

\subsection*{Proposed enhancement}
Model fill probability based on queue priority:
\begin{itemize}
  \item Track prior ask depth over time.
  \item Assign lower fill probability to passive orders deeper in the queue.
  \item Introduce slippage for aggressive orders due to short-term price movements.
\end{itemize}

This would make the simulator more realistic by accounting for latency, competition, and adverse selection, and allow dynamic adjustment of the \texttt{theta\_queue} penalty.

\section*{How to Run}

\begin{lstlisting}[language=bash]
python3 backtest.py
\end{lstlisting}

\textbf{Outputs:}
\begin{itemize}
  \item JSON summary of costs and optimal parameters
  \item \texttt{results.png}: cumulative cost plot of all strategies
\end{itemize}

\end{document}
