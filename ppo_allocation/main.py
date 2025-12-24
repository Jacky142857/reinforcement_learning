import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# Stable Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Import reward functions
from reward_functions import get_all_reward_functions, get_reward_function_by_name, RewardFunction

SEED = 42


def set_global_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


set_global_seed(SEED)

WARMUP_STEPS = 20  # number of historical steps skipped before trading
MIN_TRAINING_TIMESTEPS = 1_000_000
MAX_TRAINING_TIMESTEPS = 5_000_000
EVAL_FREQUENCY = 50_000
EVAL_N_EPISODES = 5
NO_IMPROVEMENT_PATIENCE = 10
MAX_SAVED_LOG_TIMESTEPS = 100


class PortfolioGymEnv(gym.Env):
    """Gymnasium-compatible portfolio environment for Stable Baselines3"""
    
    def __init__(self, stock_data_dict, reward_function, initial_balance=100000, risk_free_rate=0.03, seed=SEED):
        super().__init__()
        
        self.stock_data_dict = stock_data_dict
        self.stock_symbols = list(stock_data_dict.keys())
        self.num_stocks = len(self.stock_symbols)
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.reward_function = reward_function
        self.seed = seed
        
        # Calculate features for all stocks
        self.features_dict = {}
        for symbol, data in stock_data_dict.items():
            features = self._calculate_features(data)
            self.features_dict[symbol] = features
        
        # Define observation and action spaces
        current_features_size = self.num_stocks * 5
        additional_size = self.num_stocks + 3
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(current_features_size + additional_size,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0, 
            shape=(self.num_stocks + 1,),
            dtype=np.float32
        )

        self.action_space.seed(self.seed)
        self.observation_space.seed(self.seed)
        
        self.reset(seed=self.seed)
    
    def _calculate_features(self, data):
        """Calculate 5 features per timestep"""
        df = data.copy()
        
        prices = df['Close'].values
        simple_returns = df['Close'].pct_change().fillna(0).values
        
        df['simple_ret'] = simple_returns
        df['short_ma'] = df['Close'].rolling(10).mean()
        df['med_ma'] = df['Close'].rolling(20).mean()
        df['volatility'] = df['Close'].pct_change().rolling(20).std().fillna(0)
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        
        features = np.column_stack([
            df['simple_ret'].values,
            (df['Close'] / df['short_ma'] - 1).fillna(0).values,
            (df['Close'] / df['med_ma'] - 1).fillna(0).values,
            df['volatility'].values,
            (df['rsi'].values - 50) / 50
        ])
        
        return features
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is None:
            seed = self.seed
        else:
            self.seed = seed
        
        super().reset(seed=seed)
        
        min_data_len = min(len(df) for df in self.stock_data_dict.values())
        self.num_steps = min_data_len

        if self.num_steps <= WARMUP_STEPS:
            raise ValueError(f"Insufficient data: num_steps={self.num_steps}")

        self.current_step = WARMUP_STEPS
        self.portfolio_value = self.initial_balance
        self.cash_allocation = 1.0
        self.stock_allocations = np.zeros(self.num_stocks)
        
        # Reset reward function state
        self.reward_function.reset()
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get state with current timestep features"""
        if self.current_step >= self.num_steps:
            state_size = self.num_stocks * 5 + self.num_stocks + 3
            return np.zeros(state_size, dtype=np.float32)
        
        current_features = []
        for symbol in self.stock_symbols:
            features_at_t = self.features_dict[symbol][self.current_step, :]
            current_features.extend(features_at_t)
        
        additional_features = np.concatenate([
            self.stock_allocations,
            [self.cash_allocation],
            [np.sum(self.stock_allocations)],
            [self.portfolio_value / self.initial_balance]
        ])
        
        final_state = np.concatenate([current_features, additional_features])
        return np.nan_to_num(final_state, 0).astype(np.float32)
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        if self.current_step >= self.num_steps - 1:
            return self._get_state(), 0, True, False, {}
        
        # Convert raw action to portfolio weights using softmax
        allocation_weights = np.exp(action) / np.sum(np.exp(action))
        
        new_stock_allocations = allocation_weights[:self.num_stocks]
        new_cash_allocation = allocation_weights[self.num_stocks]
        
        allocation_changes = np.abs(new_stock_allocations - self.stock_allocations).sum()
        allocation_changes += np.abs(new_cash_allocation - self.cash_allocation)
        
        prev_portfolio_value = self.portfolio_value
        
        self.stock_allocations = new_stock_allocations.copy()
        self.cash_allocation = new_cash_allocation
        
        self.current_step += 1
        
        # Calculate portfolio return
        if self.current_step < self.num_steps:
            daily_risk_free_rate = (1 + self.risk_free_rate) ** (1/252) - 1
            cash_return = daily_risk_free_rate
            
            stock_simple_returns = np.array([
                self.features_dict[symbol][self.current_step, 0]
                for symbol in self.stock_symbols
            ])
            
            portfolio_simple_return = (self.cash_allocation * cash_return +
                                     np.sum(self.stock_allocations * stock_simple_returns))
            
            self.portfolio_value *= (1 + portfolio_simple_return)
            
            # Use the reward function
            reward = self.reward_function.compute(
                portfolio_simple_return=portfolio_simple_return,
                portfolio_value=self.portfolio_value,
                prev_portfolio_value=prev_portfolio_value,
                initial_balance=self.initial_balance,
                stock_allocations=self.stock_allocations,
                cash_allocation=self.cash_allocation,
                stock_simple_returns=stock_simple_returns,
                allocation_changes=allocation_changes,
                num_stocks=self.num_stocks
            )
        else:
            reward = 0
            portfolio_simple_return = 0
        
        done = self.current_step >= self.num_steps - 1
        truncated = False
        info = {
            'portfolio_value': self.portfolio_value,
            'allocations': allocation_weights,
            'return': portfolio_simple_return,
            'allocation_changes': allocation_changes
        }
        
        return self._get_state(), reward, done, truncated, info


class PortfolioCallback(BaseCallback):
    """Callback to track portfolio performance during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_return = 0
        self.current_episode_length = 0
    
    def _on_step(self):
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_return += self.locals['rewards'][0]
            self.current_episode_length += 1
        
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            
            if self.episode_count % 10 == 0:
                avg_return = np.mean(self.episode_returns[-10:])
                print(f"Episode {self.episode_count}: "
                      f"Return={self.current_episode_return:.6f}, "
                      f"Avg Return (last 10): {avg_return:.6f}")
            
            self.current_episode_return = 0
            self.current_episode_length = 0
        
        return True


class LossLoggingCallback(BaseCallback):
    """Callback that records loss metrics to a text file during training."""

    _POLICY_KEYS = ("train/policy_gradient_loss", "train/policy_loss", "train/actor_loss")
    _VALUE_KEYS = ("train/value_loss", "train/qf_loss", "train/critic_loss")
    _ENTROPY_KEYS = ("train/entropy_loss", "train/entropy")
    _TOTAL_KEYS = ("train/loss",)

    def __init__(self, log_path, max_records=MAX_SAVED_LOG_TIMESTEPS, total_timesteps=None, verbose=0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.records = []
        self.max_records = max_records
        self.total_timesteps = total_timesteps
        self._record_interval = None
        if self.max_records:
            if self.total_timesteps:
                self._record_interval = max(1, self.total_timesteps // self.max_records)
            else:
                self._record_interval = None

    def _maybe_to_float(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[-1]
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fetch_from_log_dict(self, log_dict, keys):
        for key in keys:
            if key in log_dict:
                val = self._maybe_to_float(log_dict[key])
                if val is not None:
                    return val
        return None

    def _append_record(self, step, policy_loss, value_loss, entropy_loss, total_loss, force=False):
        step = int(step)
        record = (step, policy_loss, value_loss, entropy_loss, total_loss)

        if self.records and self.records[-1][0] == step:
            self.records[-1] = record
            return

        if not force and self.max_records is not None:
            if self._record_interval:
                if self.records:
                    last_step = self.records[-1][0]
                    if step - last_step < self._record_interval:
                        return
                if len(self.records) >= self.max_records:
                    return
            elif len(self.records) >= self.max_records:
                return

        if force and self.max_records is not None and len(self.records) >= self.max_records:
            self.records[-1] = record
        else:
            self.records.append(record)

    def _value_to_str(self, value):
        if value is None:
            return "nan"
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return f"{value:.10f}"

    def _write_records(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        header = "timesteps,policy_loss(actor_loss),value_loss(critic_loss),entropy_loss,total_loss"
        with open(self.log_path, "w") as f:
            f.write(header + "\n")
            for step, policy_loss, value_loss, entropy_loss, total_loss in self.records:
                f.write(
                    f"{step},"
                    f"{self._value_to_str(policy_loss)},"
                    f"{self._value_to_str(value_loss)},"
                    f"{self._value_to_str(entropy_loss)},"
                    f"{self._value_to_str(total_loss)}\n"
                )

    def _ensure_final_record(self, log_dict):
        policy_loss = self._fetch_from_log_dict(log_dict, self._POLICY_KEYS)
        value_loss = self._fetch_from_log_dict(log_dict, self._VALUE_KEYS)
        entropy_loss = self._fetch_from_log_dict(log_dict, self._ENTROPY_KEYS)
        total_loss = self._fetch_from_log_dict(log_dict, self._TOTAL_KEYS)
        if total_loss is None and all(v is not None for v in (policy_loss, value_loss, entropy_loss)):
            total_loss = policy_loss + value_loss + entropy_loss

        if any(v is not None for v in (policy_loss, value_loss, entropy_loss, total_loss)):
            self._append_record(self.num_timesteps, policy_loss, value_loss, entropy_loss, total_loss, force=True)

    def _on_step(self):
        log_dict = getattr(self.model.logger, "name_to_value", None)
        if not log_dict:
            return True

        policy_loss = self._fetch_from_log_dict(log_dict, self._POLICY_KEYS)
        value_loss = self._fetch_from_log_dict(log_dict, self._VALUE_KEYS)
        entropy_loss = self._fetch_from_log_dict(log_dict, self._ENTROPY_KEYS)
        total_loss = self._fetch_from_log_dict(log_dict, self._TOTAL_KEYS)

        if policy_loss is None and value_loss is None and entropy_loss is None and total_loss is None:
            return True

        if total_loss is None and all(v is not None for v in (policy_loss, value_loss, entropy_loss)):
            total_loss = policy_loss + value_loss + entropy_loss

        self._append_record(self.num_timesteps, policy_loss, value_loss, entropy_loss, total_loss)
        return True

    def _on_training_end(self):
        log_dict = getattr(self.model.logger, "name_to_value", None) or {}
        if not self.records:
            self._ensure_final_record(log_dict)
        elif log_dict:
            self._ensure_final_record(log_dict)

        if not self.records:
            # Create an empty file with header for consistency.
            self._append_record(0, None, None, None, None, force=True)

        self._write_records()


def prepare_data(df):
    """Prepare stock data with forward-fill only"""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date').reset_index(drop=True)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            df[col] = df['Close'] if 'Close' in df.columns else df.iloc[:, 1]
    
    for col in required_columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill()
    
    initial_nans = df[required_columns].isnull().any(axis=1)
    if initial_nans.any():
        first_valid_idx = (~initial_nans).idxmax()
        df = df.loc[first_valid_idx:].reset_index(drop=True)
    
    return df


def load_all_stock_data(stock_symbols):
    """Load all stock data"""
    stock_data_dict = {}
    
    # Get the parent directory (go up from ppo_allocation to root)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    stock_data_dir = root_dir / 'stock_data'
    
    for symbol in stock_symbols:
        file_path = stock_data_dir / f'{symbol}_data.csv'
        try:
            data_df = pd.read_csv(file_path)
            data_df['Date'] = pd.to_datetime(data_df['Date'], utc=True)
            data_df = data_df.sort_values('Date').reset_index(drop=True)
            
            stock_data_dict[symbol] = data_df
            print(f"Loaded {symbol}: {len(data_df)} rows")
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")
    
    return stock_data_dict


def calculate_buy_hold_return(stock_data_dict):
    """Calculate buy and hold return for equal weight portfolio"""
    equal_weight = 1.0 / len(stock_data_dict)
    
    stock_returns = []
    for symbol, data in stock_data_dict.items():
        if len(data) > 1:
            start_price = data.iloc[0]['Close']
            end_price = data.iloc[-1]['Close']
            stock_simple_return = (end_price - start_price) / start_price
            stock_returns.append(stock_simple_return)
    
    if not stock_returns:
        return 0.0
    
    portfolio_simple_return = sum(equal_weight * ret for ret in stock_returns)
    return portfolio_simple_return * 100


def calculate_max_drawdown(values):
    """Calculate maximum drawdown percentage for a series of values"""
    if not values:
        return 0.0
    
    values_array = np.array(values, dtype=float)
    running_max = np.maximum.accumulate(values_array)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdowns = np.where(running_max > 0, (values_array - running_max) / running_max, 0.0)
    max_drawdown = drawdowns.min() if drawdowns.size > 0 else 0.0
    return max_drawdown * 100.0


def train_sb3_portfolio(stock_data_dict, reward_function, max_timesteps=MAX_TRAINING_TIMESTEPS,
                        split_ratio=0.8, seed=SEED, loss_log_path=None):
    """Train PPO agent using Stable Baselines3"""
    
    # Split data
    train_data_dict = {}
    test_data_dict = {}
    
    print(f"\nSplitting data: {split_ratio*100:.0f}% training, {(1-split_ratio)*100:.0f}% testing")
    print("-" * 60)
    
    for symbol, data in stock_data_dict.items():
        split_idx = int(len(data) * split_ratio)
        
        train_raw = data[:split_idx].reset_index(drop=True)
        test_raw = data[split_idx:].reset_index(drop=True)
        
        train_data_dict[symbol] = prepare_data(train_raw)
        test_data_dict[symbol] = prepare_data(test_raw)
        
        print(f"{symbol}: {len(data)} total → {len(train_data_dict[symbol])} train, {len(test_data_dict[symbol])} test")
    
    # Create environment with Monitor wrapper
    def make_env():
        env = PortfolioGymEnv(train_data_dict, reward_function, seed=seed)
        env = Monitor(env)
        return env
    
    train_env = DummyVecEnv([make_env])
    train_env.seed(seed)
    train_env.reset()
    
    def make_eval_env():
        eval_seed = seed + 10_000
        env = PortfolioGymEnv(train_data_dict, reward_function, seed=eval_seed)
        env = Monitor(env)
        return env
    
    eval_env = DummyVecEnv([make_eval_env])
    eval_env.seed(seed + 10_000)
    eval_env.reset()
    
    if max_timesteps < MIN_TRAINING_TIMESTEPS:
        raise ValueError(
            f"max_timesteps ({max_timesteps}) must be >= MIN_TRAINING_TIMESTEPS ({MIN_TRAINING_TIMESTEPS})"
        )
    
    callback = PortfolioCallback()
    loss_callback = (
        LossLoggingCallback(loss_log_path, total_timesteps=max_timesteps)
        if loss_log_path
        else None
    )
    
    print(f"\nTraining with reward function: {reward_function.name}")
    print(f"Training timesteps: min={MIN_TRAINING_TIMESTEPS:,} max={max_timesteps:,}")
    print(f"Early stopping patience (evals): {NO_IMPROVEMENT_PATIENCE}")
    print(f"Evaluation frequency: every {EVAL_FREQUENCY:,} steps ({EVAL_N_EPISODES} episodes)")
    print("-" * 60)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_portfolio_tensorboard/",
        seed=seed
    )
    
    min_evals_before_stop = max(1, math.ceil(MIN_TRAINING_TIMESTEPS / EVAL_FREQUENCY))
    early_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=NO_IMPROVEMENT_PATIENCE,
        min_evals=min_evals_before_stop,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQUENCY,
        n_eval_episodes=EVAL_N_EPISODES,
        deterministic=True,
        callback_after_eval=early_stop_callback,
        verbose=1
    )
    
    combined_callbacks = [callback, eval_callback]
    if loss_callback is not None:
        combined_callbacks.append(loss_callback)
    
    # Train with early stopping
    model.learn(
        total_timesteps=max_timesteps,
        callback=combined_callbacks,
        progress_bar=True
    )
    training_timesteps = model.num_timesteps
    
    train_buy_hold_return = calculate_buy_hold_return(train_data_dict)
    
    return model, callback.episode_returns, test_data_dict, train_buy_hold_return, training_timesteps


def test_sb3_portfolio(model, test_data_dict, reward_function, seed=SEED):
    """Test trained SB3 model"""
    test_env = PortfolioGymEnv(test_data_dict, reward_function, seed=seed)
    
    obs, _ = test_env.reset(seed=seed)
    portfolio_values = [test_env.portfolio_value]
    allocations_history = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        allocations_history.append({
            'step': test_env.current_step,
            'cash': test_env.cash_allocation,
            'stocks': test_env.stock_allocations.copy()
        })
        
        if truncated:
            break
    
    return portfolio_values, allocations_history, list(test_data_dict.keys())


def plot_results(results, reward_name, output_dir):
    """Plot training and testing results as separate images in a plots folder"""
    
    # Create plots subdirectory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Calculate buy & hold values once for reuse
    buy_hold_values = None
    if 'test_data_dict' in results:
        test_data_dict = results['test_data_dict']
        warmup = results.get('warmup', WARMUP_STEPS)
        equal_weight = 1.0 / len(test_data_dict) if test_data_dict else 0.0
        
        initial_value = results['portfolio_values'][0]
        buy_hold_values = [initial_value]
        
        for t in range(1, len(results['portfolio_values'])):
            dataset_idx_prev = warmup + t - 1
            dataset_idx_curr = warmup + t
            
            step_portfolio_return = 0.0
            for _, data in test_data_dict.items():
                if dataset_idx_curr >= len(data) or dataset_idx_prev >= len(data):
                    continue
                
                prev_price = data.iloc[dataset_idx_prev]['Close']
                curr_price = data.iloc[dataset_idx_curr]['Close']
                
                if prev_price > 0:
                    stock_return = (curr_price - prev_price) / prev_price
                else:
                    stock_return = 0.0
                
                step_portfolio_return += equal_weight * stock_return
            
            new_value = buy_hold_values[-1] * (1 + step_portfolio_return)
            buy_hold_values.append(new_value)
    
    # ========================================================================
    # PLOT 1: Training Returns
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if len(results['training_returns']) > 0:
        episodes = range(1, len(results['training_returns']) + 1)
        ax.plot(episodes, results['training_returns'], alpha=0.3, color='lightblue', label='Episode Returns')
        
        if len(results['training_returns']) > 20:
            window = min(50, len(results['training_returns']) // 10)
            ma = pd.Series(results['training_returns']).rolling(window).mean()
            ax.plot(episodes, ma, color='darkblue', linewidth=2, label=f'{window}-Episode MA')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'Training Returns Over Episodes\n({reward_name})', fontweight='bold', fontsize=14)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Returns', fontweight='bold')
    
    plt.tight_layout()
    plot1_path = plots_dir / "1_training_returns.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 2: Portfolio Value with Buy & Hold (Overlapped)
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    steps = range(len(results['portfolio_values']))
    ax.plot(steps, results['portfolio_values'], color='green', linewidth=2.5, label='PPO Agent', zorder=3)
    
    if buy_hold_values is not None:
        ax.plot(range(len(buy_hold_values)), buy_hold_values, color='orange', 
                linewidth=2.5, linestyle='--', label='Buy & Hold (Equal Weight)', zorder=2)
    
    ax.set_title(f'Portfolio Performance - Testing (Comparison)\n({reward_name})', fontweight='bold', fontsize=14)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plot2_path = plots_dir / "2_portfolio_comparison.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 3: Allocations
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    if results['allocations_history']:
        steps = [a['step'] for a in results['allocations_history']]
        cash = [a['cash'] for a in results['allocations_history']]
        stocks = np.array([a['stocks'] for a in results['allocations_history']])
        
        stock_symbols = results.get('stock_symbols', [f'Stock {i+1}' for i in range(stocks.shape[1])])
        labels = ['Cash'] + stock_symbols
        
        ax.stackplot(steps, cash, *stocks.T, labels=labels, alpha=0.7)
        ax.set_title(f'Portfolio Allocation Over Time\n({reward_name})', fontweight='bold', fontsize=14)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Allocation Weight', fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot3_path = plots_dir / "3_allocations.png"
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 4: Performance Comparison Bar Chart
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    categories = ['Buy & Hold\n(Test)', 'PPO Agent\n(Test)']
    returns = [results['test_buy_hold_return'], results['total_return']]
    colors = ['orange', 'green']
    
    bars = ax.bar(categories, returns, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_title(f'Performance Comparison\n({reward_name})', fontweight='bold', fontsize=14)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in enumerate(returns):
        height = bars[bar].get_height()
        ax.text(bars[bar].get_x() + bars[bar].get_width()/2., height + (1 if height >= 0 else -2),
                f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=12)
    
    outperformance = results['total_return'] - results['test_buy_hold_return']
    color = 'green' if outperformance > 0 else 'red'
    ax.text(0.5, 0.02, f'Outperformance: {outperformance:+.2f}%', 
            transform=ax.transAxes, ha='center', fontsize=12, 
            fontweight='bold', color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot4_path = plots_dir / "4_performance_bars.png"
    plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 5: PPO Agent Only (Without Buy & Hold)
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    steps = range(len(results['portfolio_values']))
    ax.plot(steps, results['portfolio_values'], color='green', linewidth=2.5, label='PPO Agent')
    
    # Add performance statistics as text
    initial_value = results['portfolio_values'][0]
    final_value = results['portfolio_values'][-1]
    pct_return = results['total_return']
    
    stats_text = f"Initial: ${initial_value:,.0f}\nFinal: ${final_value:,.0f}\nReturn: {pct_return:.2f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'PPO Agent Performance - Testing\n({reward_name})', fontweight='bold', fontsize=14)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plot5_path = plots_dir / "5_ppo_only.png"
    plt.savefig(plot5_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 5 plots saved in: {plots_dir}")
    print(f"   - 1_training_returns.png")
    print(f"   - 2_portfolio_comparison.png")
    print(f"   - 3_allocations.png")
    print(f"   - 4_performance_bars.png")
    print(f"   - 5_ppo_only.png")


def save_results_summary(results, reward_name, output_dir):
    """Save results summary to a text file"""
    summary_path = output_dir / f"{reward_name}_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"RESULTS SUMMARY: {reward_name}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Reward Function: {reward_name}\n")
        f.write(f"PPO Return: {results['total_return']:.2f}%\n")
        f.write(f"Buy & Hold Return: {results['test_buy_hold_return']:.2f}%\n")
        f.write(f"Outperformance: {results['total_return'] - results['test_buy_hold_return']:.2f}%\n\n")
        
        f.write(f"Initial Portfolio Value: ${results['portfolio_values'][0]:,.2f}\n")
        f.write(f"Final Portfolio Value: ${results['portfolio_values'][-1]:,.2f}\n")
        f.write(f"Number of Training Episodes: {len(results['training_returns'])}\n")
        if 'training_timesteps' in results:
            f.write(f"Training Timesteps Used: {results['training_timesteps']:,}\n")
        
        warmup_steps = results.get('warmup', WARMUP_STEPS)
        trading_steps = max(0, len(results['portfolio_values']) - 1)
        f.write(f"Warmup Steps (no trading): {warmup_steps}\n")
        f.write(f"Trading Steps Evaluated: {trading_steps}\n\n")
        
        max_drawdown_pct = calculate_max_drawdown(results.get('portfolio_values', []))
        f.write(f"Max Drawdown (PPO): {max_drawdown_pct:.2f}%\n\n")
        
        if len(results['training_returns']) > 0:
            f.write(f"Average Training Return: {np.mean(results['training_returns']):.6f}\n")
            f.write(f"Training Return Std Dev: {np.std(results['training_returns']):.6f}\n")
    
    print(f"📝 Summary saved as: {summary_path}")


def run_experiment_for_reward(stock_data_dict, reward_function, output_dir,
                              max_timesteps=MAX_TRAINING_TIMESTEPS, seed=SEED):
    """Run a complete experiment for a single reward function"""
    
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {reward_function.name}")
    print("="*80)

    set_global_seed(seed)
    
    # Create subdirectory for this reward
    reward_dir = output_dir / reward_function.name.replace(' ', '_').replace('/', '_')
    reward_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = reward_dir / "training_losses.txt"
    
    # Train
    model, training_returns, test_data_dict, train_bh, train_timesteps = train_sb3_portfolio(
        stock_data_dict,
        reward_function,
        max_timesteps=max_timesteps,
        split_ratio=0.8,
        seed=seed,
        loss_log_path=loss_log_path
    )
    
    # Test
    print(f"\nTesting model for {reward_function.name}...")
    portfolio_values, allocations, stock_symbols = test_sb3_portfolio(
        model,
        test_data_dict,
        reward_function,
        seed=seed
    )
    
    warmup = WARMUP_STEPS
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    equal_weight = 1.0 / len(test_data_dict) if test_data_dict else 0.0
    bh_value = initial_value
    bh_values = [bh_value]
    
    for t in range(1, len(portfolio_values)):
        dataset_idx_prev = warmup + t - 1
        dataset_idx_curr = warmup + t
        
        step_return = 0.0
        for _, data in test_data_dict.items():
            if dataset_idx_curr >= len(data) or dataset_idx_prev >= len(data):
                continue
            
            prev_p = data.iloc[dataset_idx_prev]['Close']
            curr_p = data.iloc[dataset_idx_curr]['Close']
            
            if prev_p > 0:
                r = (curr_p - prev_p) / prev_p
            else:
                r = 0.0
            
            step_return += equal_weight * r
        
        bh_value = bh_value * (1 + step_return)
        bh_values.append(bh_value)
    
    final_bh_value = bh_values[-1]
    bh_return_pct = (final_bh_value - bh_values[0]) / bh_values[0] * 100.0
    test_bh = bh_return_pct
    
    results = {
        'training_returns': training_returns,
        'portfolio_values': portfolio_values,
        'allocations_history': allocations,
        'training_timesteps': train_timesteps,
        'total_return': total_return,
        'test_buy_hold_return': test_bh,
        'train_buy_hold_return': train_bh,
        'test_data_dict': test_data_dict,
        'stock_symbols': stock_symbols,
        'warmup': warmup
    }
    
    print(f"\nResults for {reward_function.name}:")
    print(f"  PPO Return: {total_return:.2f}%")
    print(f"  Buy & Hold: {test_bh:.2f}%")
    print(f"  Outperformance: {total_return - test_bh:.2f}%")
    
    # Save results
    plot_results(results, reward_function.name, reward_dir)
    save_results_summary(results, reward_function.name, reward_dir)
    
    # Save model
    model_path = reward_dir / "model"
    model.save(str(model_path))
    print(f"💾 Model saved as: {model_path}.zip")
    
    return results


def display_reward_menu():
    """Display interactive menu for selecting reward functions"""
    reward_functions = get_all_reward_functions()
    
    print("\n" + "="*80)
    print("AVAILABLE REWARD FUNCTIONS")
    print("="*80)
    
    # Group rewards into categories
    standard_rewards = []
    active_rewards = []
    
    for i, reward in enumerate(reward_functions):
        if any(keyword in reward.name for keyword in ['Trading', 'Momentum', 'Timing', 'Regime', 
                                                       'Turnover', 'Tactical', 'Opportunity', 
                                                       'Rebalancing', 'Strength', 'Risk-On']):
            active_rewards.append((i, reward))
        else:
            standard_rewards.append((i, reward))
    
    print("\n📊 STANDARD REWARDS (Similar to Buy & Hold):")
    print("-" * 80)
    for i, reward in standard_rewards:
        print(f"  {i+1:2d}. {reward.name}")
    
    print("\n⚡ ACTIVE TRADING REWARDS (Anti-Buy-and-Hold):")
    print("-" * 80)
    for i, reward in active_rewards:
        print(f"  {i+1:2d}. {reward.name}")
    
    print("\n" + "="*80)
    print("OPTIONS:")
    print("  - Enter a number (1-22) to run a specific reward")
    print("  - Enter 'all' to run all rewards sequentially")
    print("  - Enter 'quit' or 'q' to exit")
    print("="*80)
    
    return reward_functions


def main():
    """Main function with interactive reward selection"""
    print("="*80)
    print("PORTFOLIO TRADING WITH STABLE BASELINES3")
    print("INTERACTIVE REWARD FUNCTION TESTING")
    print("="*80)
    print("Configuration:")
    print("  - NO transaction costs")
    print("  - NO reward clipping")
    print("  - Cash earns 3% annual interest (risk-free rate)")
    print("="*80)
    
    # Setup directories
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    
    # Stock symbols
    stock_symbols = [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO",
        "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH",
        "V", "VZ", "WBA", "WMT", "RTX"
    ]
    
    print(f"\nLoading data for {len(stock_symbols)} stocks...")
    stock_data_dict = load_all_stock_data(stock_symbols)
    
    if len(stock_data_dict) == 0:
        print("❌ No stock data loaded. Please check stock_data directory.")
        return
    
    # Interactive loop
    all_results = {}
    
    while True:
        reward_functions = display_reward_menu()
        
        user_input = input("\n👉 Enter your choice: ").strip().lower()
        
        if user_input in ['quit', 'q', 'exit']:
            print("\n👋 Exiting. Goodbye!")
            break
        
        elif user_input == 'all':
            print(f"\n🚀 Running ALL {len(reward_functions)} reward functions...")
            confirm = input("This will take a long time. Continue? (yes/no): ").strip().lower()
            
            if confirm not in ['yes', 'y']:
                print("Cancelled.")
                continue
            
            for i, reward_func in enumerate(reward_functions, 1):
                print(f"\n{'='*80}")
                print(f"EXPERIMENT {i}/{len(reward_functions)}")
                print(f"{'='*80}")
                
                try:
                    results = run_experiment_for_reward(
                        stock_data_dict,
                        reward_func,
                        output_dir,
                        max_timesteps=MAX_TRAINING_TIMESTEPS
                    )
                    all_results[reward_func.name] = results
                except Exception as e:
                    print(f"❌ Error running experiment for {reward_func.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Create comparison summary
            print("\n" + "="*80)
            print("FINAL COMPARISON ACROSS ALL REWARD FUNCTIONS")
            print("="*80)
            
            comparison_path = output_dir / "comparison_summary.txt"
            with open(comparison_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("REWARD FUNCTION COMPARISON\n")
                f.write("="*80 + "\n\n")
                
                sorted_results = sorted(
                    all_results.items(),
                    key=lambda x: x[1]['total_return'] - x[1]['test_buy_hold_return'],
                    reverse=True
                )
                
                f.write(f"{'Rank':<6} {'Reward Function':<40} {'PPO Return':<12} {'B&H Return':<12} {'Outperf':<10}\n")
                f.write("-"*80 + "\n")
                
                for rank, (name, results) in enumerate(sorted_results, 1):
                    ppo_ret = results['total_return']
                    bh_ret = results['test_buy_hold_return']
                    outperf = ppo_ret - bh_ret
                    
                    f.write(f"{rank:<6} {name:<40} {ppo_ret:>10.2f}% {bh_ret:>10.2f}% {outperf:>+9.2f}%\n")
                    
                    print(f"{rank:<6} {name:<40} PPO: {ppo_ret:>7.2f}%  B&H: {bh_ret:>7.2f}%  Δ: {outperf:>+7.2f}%")
            
            print(f"\n📊 Comparison summary saved as: {comparison_path}")
            print("\n✅ All experiments completed!")
        
        elif user_input.isdigit():
            choice = int(user_input)
            
            if 1 <= choice <= len(reward_functions):
                selected_reward = reward_functions[choice - 1]
                
                print(f"\n🎯 Selected: {selected_reward.name}")
                
                try:
                    results = run_experiment_for_reward(
                        stock_data_dict,
                        selected_reward,
                        output_dir,
                        max_timesteps=MAX_TRAINING_TIMESTEPS
                    )
                    all_results[selected_reward.name] = results
                    
                    print(f"\n✅ Experiment completed for: {selected_reward.name}")
                    
                except Exception as e:
                    print(f"❌ Error running experiment: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(reward_functions)}.")
        
        else:
            print("❌ Invalid input. Please enter a number, 'all', or 'quit'.")
        
        # Ask if user wants to continue
        if user_input != 'all':
            continue_choice = input("\n🔄 Run another experiment? (yes/no): ").strip().lower()
            if continue_choice not in ['yes', 'y']:
                print("\n👋 Exiting. Goodbye!")
                break
    
    # Final summary if any experiments were run
    if all_results:
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        print(f"Total experiments completed: {len(all_results)}")
        print(f"Results saved in: {output_dir.absolute()}")
        
        if len(all_results) > 1:
            print("\n📊 Performance Ranking:")
            sorted_results = sorted(
                all_results.items(),
                key=lambda x: x[1]['total_return'] - x[1]['test_buy_hold_return'],
                reverse=True
            )
            
            for rank, (name, results) in enumerate(sorted_results, 1):
                ppo_ret = results['total_return']
                bh_ret = results['test_buy_hold_return']
                outperf = ppo_ret - bh_ret
                print(f"  {rank}. {name}: {outperf:+.2f}% outperformance")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
