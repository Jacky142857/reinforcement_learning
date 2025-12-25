# Deep Reinforcement Learning for Portfolio Allocation

This project presents a comprehensive empirical study on the impact of reward function design on the performance and stability of Deep Reinforcement Learning (DRL) agents in portfolio allocation.

Addressing common methodological limitations in existing literature—such as selection bias and the lack of cash management options—this study utilizes the Dow Jones 30 constituent stocks as the asset universe and explicitly incorporates a cash asset to allow for risk-off positioning.

## Algorithms
We benchmark three prominent actor-critic algorithms across twenty-two distinct reward formulations:
*   **PPO** (Proximal Policy Optimization)
*   **SAC** (Soft Actor-Critic)
*   **TD3** (Twin Delayed Deep Deterministic Policy Gradient)

## Project Structure

```
.
├── ppo_allocation/     # PPO implementation and results
│   ├── main.py         # Main entry point for PPO training/testing
│   ├── reward_functions.py # Definitions of reward functions
│   └── output/         # Results, plots, and models for PPO
├── sac_allocation/     # SAC implementation and results
│   ├── main.py         # Main entry point for SAC
│   └── ...
├── td3_allocation/     # TD3 implementation and results
│   ├── main.py         # Main entry point for TD3
│   └── ...
├── analysis/           # Analysis and visualization scripts
│   ├── aggregate_results.py # Aggregates metrics from all runs
│   ├── color_analysis.py    # Analyzes allocation colors/patterns
│   └── results.csv          # Consolidated results file
├── stock_data/         # Historical stock data (CSV format)
└── .venv/              # Python virtual environment
```

## Setup and Installation

1.  **Prerequisites**: Ensure you have Python 3.10+ installed.
2.  **Environment**: It is recommended to use the provided virtual environment or create a new one.

    ```bash
    # Activate existing venv (if available)
    source .venv/bin/activate
    
    # OR create and activate a new one
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Dependencies**: Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, key dependencies include `torch`, `gymnasium`, `stable-baselines3`, `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.)*

## Usage

Each algorithm has its own directory and main script. The scripts feature an interactive menu to select reward functions.

### Running PPO
```bash
python ppo_allocation/main.py
```

### Running SAC
```bash
python sac_allocation/main.py
```

### Running TD3
```bash
python td3_allocation/main.py
```

**Interactive Menu Options:**
*   Select a specific reward function number (1-22).
*   Type `all` to run experiments for all reward functions sequentially.
*   Type `q` or `quit` to exit.

## Reward Functions

The project evaluates 22 reward functions, categorized into:
*   **Standard Rewards**: Emphasize raw returns or simple risk adjustments (similar to Buy & Hold).
*   **Active Trading Rewards**: Incentivize active management, rebalancing, and risk mitigation (Anti-Buy-and-Hold).

Examples include:
*   `Raw Portfolio Return`
*   `Sharpe Ratio`
*   `Sortino Ratio`
*   `Max Drawdown Penalty`
*   `Volatility Penalty`

## Analysis

After running experiments, you can analyze the results using the scripts in the `analysis/` directory.

### Aggregate Results
To compile metrics from all trained agents into a single CSV or DataFrame:
```bash
python analysis/aggregate_results.py --to-csv analysis/results.csv
```

### Visualization
Various plotting scripts are available in `analysis/` to generate figures for:
*   Efficient frontiers (`plot_frontier.py`)
*   Allocation patterns (`plot_color.py`)
*   Convergence analysis (`sharpe_convergence.png`, etc.)

## Data
The `stock_data/` directory contains historical price data (Open, High, Low, Close, Volume) for the tickers used in the environment (DJ30 components).