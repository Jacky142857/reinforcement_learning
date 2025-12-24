"""
Reward function definitions for portfolio optimization.
Each function takes standardized inputs and returns a reward value.
"""

import numpy as np


class RewardFunction:
    """Base class for reward functions with state tracking"""
    
    def __init__(self, name):
        self.name = name
        self.state = {}  # For storing persistent state across steps
    
    def reset(self):
        """Reset any internal state"""
        self.state = {}
    
    def compute(self, **kwargs):
        """Compute reward - to be implemented by subclasses"""
        raise NotImplementedError


# ============================================================================
# STANDARD REWARDS (Similar to Buy & Hold)
# ============================================================================

class RawPortfolioReturn(RewardFunction):
    """Option 1: Raw Portfolio Return"""
    
    def __init__(self):
        super().__init__("Raw Portfolio Return")
    
    def compute(self, portfolio_simple_return, **kwargs):
        return portfolio_simple_return


class LogReturn(RewardFunction):
    """Option 2: Log Return (smoother, handles large swings better)"""
    
    def __init__(self):
        super().__init__("Log Return")
    
    def compute(self, portfolio_value, prev_portfolio_value, **kwargs):
        return np.log(portfolio_value / prev_portfolio_value)


class ReturnOverRollingVolatility(RewardFunction):
    """Option 3: Sharpe-like Reward (return adjusted by volatility)"""
    
    def __init__(self):
        super().__init__("Sharpe-like Reward")
    
    def reset(self):
        """Reset state and reinitialize returns_history"""
        super().reset()
        self.state['returns_history'] = []
    
    def compute(self, portfolio_simple_return, **kwargs):
        if 'returns_history' not in self.state:
            self.state['returns_history'] = []
            
        self.state['returns_history'].append(portfolio_simple_return)
        if len(self.state['returns_history']) > 20:
            self.state['returns_history'].pop(0)
        
        volatility = np.std(self.state['returns_history']) if len(self.state['returns_history']) > 1 else 1.0
        return portfolio_simple_return / (volatility + 1e-8)


class OnlineSharpeRatioReward(RewardFunction):
    """Option 4: Differential Sharpe Ratio (online Sharpe computation)"""
    
    def __init__(self):
        super().__init__("Differential Sharpe Ratio")
    
    def reset(self):
        """Reset state"""
        super().reset()
        self.state['avg_return'] = 0.0
        self.state['avg_return_sq'] = 0.0
        self.state['sharpe_count'] = 0
    
    def compute(self, portfolio_simple_return, **kwargs):
        if 'sharpe_count' not in self.state:
            self.state['avg_return'] = 0.0
            self.state['avg_return_sq'] = 0.0
            self.state['sharpe_count'] = 0
            
        self.state['sharpe_count'] += 1
        delta = portfolio_simple_return - self.state['avg_return']
        self.state['avg_return'] += delta / self.state['sharpe_count']
        self.state['avg_return_sq'] += delta * (portfolio_simple_return - self.state['avg_return'])
        variance = self.state['avg_return_sq'] / max(1, self.state['sharpe_count'] - 1)
        std = np.sqrt(variance + 1e-8)
        return (portfolio_simple_return - self.state['avg_return']) / (std + 1e-8)


class NormalizedProfit(RewardFunction):
    """Option 5: Normalized Profit (scaled by initial balance)"""
    
    def __init__(self):
        super().__init__("Normalized Profit")
    
    def compute(self, portfolio_value, prev_portfolio_value, initial_balance, **kwargs):
        return (portfolio_value - prev_portfolio_value) / initial_balance


class SortinoLikeReward(RewardFunction):
    """Option 6: Sortino-like (penalize downside more than upside)"""
    
    def __init__(self):
        super().__init__("Sortino-like Reward")
    
    def compute(self, portfolio_simple_return, **kwargs):
        if portfolio_simple_return < 0:
            return 2.0 * portfolio_simple_return  # Double penalty for losses
        else:
            return portfolio_simple_return


class ReturnMinusDrawdownPenalty(RewardFunction):
    """Option 7: Return minus Drawdown Penalty"""
    
    def __init__(self):
        super().__init__("Return minus Drawdown Penalty")
    
    def reset(self):
        """Reset state"""
        super().reset()
        self.state['peak_value'] = None
    
    def compute(self, portfolio_simple_return, portfolio_value, initial_balance, **kwargs):
        if 'peak_value' not in self.state or self.state['peak_value'] is None:
            self.state['peak_value'] = initial_balance
        self.state['peak_value'] = max(self.state['peak_value'], portfolio_value)
        drawdown = (self.state['peak_value'] - portfolio_value) / self.state['peak_value']
        return portfolio_simple_return - 0.5 * drawdown


class ExcessReturnOverBenchmark(RewardFunction):
    """Option 8: Excess Return over Benchmark (equal-weight portfolio)"""
    
    def __init__(self):
        super().__init__("Excess Return over Benchmark")
    
    def compute(self, portfolio_simple_return, stock_simple_returns, **kwargs):
        equal_weight_return = np.mean(stock_simple_returns)
        return portfolio_simple_return - equal_weight_return


class RiskAdjustedByConcentration(RewardFunction):
    """Option 9: Risk-Adjusted Return (return per unit of position concentration)."""
    
    def __init__(self):
        super().__init__("Risk-Adjusted by Concentration")
    
    def compute(self, portfolio_simple_return, stock_allocations, **kwargs):
        cash_allocation = kwargs.get("cash_allocation", 0.0)
        total_allocations = np.append(stock_allocations, np.clip(cash_allocation, 0.0, 1.0))
        concentration = np.sum(total_allocations ** 2)  # Herfindahl index over all capital, including cash
        return portfolio_simple_return / (concentration + 1e-8)


class CalmarLikeReward(RewardFunction):
    """Option 10: Calmar-like (return adjusted by max drawdown)"""
    
    def __init__(self):
        super().__init__("Calmar-like Reward")
    
    def reset(self):
        """Reset state"""
        super().reset()
        self.state['peak_value'] = None
        self.state['max_drawdown'] = 0.0
    
    def compute(self, portfolio_simple_return, portfolio_value, initial_balance, **kwargs):
        if 'peak_value' not in self.state or self.state['peak_value'] is None:
            self.state['peak_value'] = initial_balance
            self.state['max_drawdown'] = 0.0
        
        self.state['peak_value'] = max(self.state['peak_value'], portfolio_value)
        current_drawdown = (self.state['peak_value'] - portfolio_value) / self.state['peak_value']
        self.state['max_drawdown'] = max(self.state['max_drawdown'], current_drawdown)
        return portfolio_simple_return / (self.state['max_drawdown'] + 0.01)


class ReturnPlusDiversificationBonus(RewardFunction):
    """Option 11: Combined Return + Diversification Bonus"""
    
    def __init__(self):
        super().__init__("Return plus Diversification Bonus")
    
    def compute(self, portfolio_simple_return, stock_allocations, num_stocks, **kwargs):
        entropy = -np.sum(stock_allocations * np.log(stock_allocations + 1e-8))
        max_entropy = np.log(num_stocks)
        diversification_bonus = 0.01 * (entropy / max_entropy)
        return portfolio_simple_return + diversification_bonus


class InformationRatioReward(RewardFunction):
    """Option 12: Information Ratio (return versus benchmark dispersion)"""
    
    def __init__(self, window=60):
        super().__init__("Information Ratio Reward")
        self.window = window
    
    def reset(self):
        """Reset stored excess returns"""
        super().reset()
        self.state['excess_returns'] = []
    
    def compute(self, portfolio_simple_return, stock_simple_returns, **kwargs):
        if 'excess_returns' not in self.state:
            self.state['excess_returns'] = []
        
        benchmark_return = np.mean(stock_simple_returns) if len(stock_simple_returns) > 0 else 0.0
        excess_return = portfolio_simple_return - benchmark_return
        
        self.state['excess_returns'].append(excess_return)
        if len(self.state['excess_returns']) > self.window:
            self.state['excess_returns'].pop(0)
        
        if len(self.state['excess_returns']) < 5:
            return excess_return
        
        mean_excess = np.mean(self.state['excess_returns'])
        tracking_error = np.std(self.state['excess_returns'])
        return mean_excess / (tracking_error + 1e-8)


class ConditionalValueAtRiskReward(RewardFunction):
    """Option 13: CVaR-adjusted Return (penalize tail risk)"""
    
    def __init__(self, alpha=0.05, window=100, penalty_scale=0.5):
        super().__init__("CVaR Reward")
        self.alpha = alpha
        self.window = window
        self.penalty_scale = penalty_scale
    
    def reset(self):
        """Reset stored returns"""
        super().reset()
        self.state['returns_history'] = []
    
    def compute(self, portfolio_simple_return, **kwargs):
        if 'returns_history' not in self.state:
            self.state['returns_history'] = []
        
        returns_history = self.state['returns_history']
        returns_history.append(portfolio_simple_return)
        if len(returns_history) > self.window:
            returns_history.pop(0)
        
        if len(returns_history) < 10:
            return portfolio_simple_return
        
        var_threshold = np.quantile(returns_history, self.alpha)
        tail_losses = [r for r in returns_history if r <= var_threshold]
        cvar = np.mean(tail_losses) if tail_losses else var_threshold
        downside_penalty = max(0.0, -cvar)
        return portfolio_simple_return - self.penalty_scale * downside_penalty


# ============================================================================
# ANTI-BUY-AND-HOLD REWARDS (Encourage Active Trading)
# ============================================================================

class TradingActivityBonus(RewardFunction):
    """Option 14: Trading Activity Bonus - Rewards profitable trades"""
    
    def __init__(self):
        super().__init__("Trading Activity Bonus")
    
    def compute(self, portfolio_simple_return, allocation_changes, **kwargs):
        if allocation_changes > 0.01:
            trading_bonus = 0.02 * allocation_changes
        else:
            trading_bonus = -0.005
        return portfolio_simple_return + trading_bonus


class MomentumExploitationReward(RewardFunction):
    """Option 15: Momentum Exploitation Reward"""
    
    def __init__(self):
        super().__init__("Momentum Exploitation Reward")
    
    def compute(self, portfolio_simple_return, stock_allocations, stock_simple_returns, **kwargs):
        winner_exposure = np.sum(stock_allocations[stock_simple_returns > 0])
        loser_exposure = np.sum(stock_allocations[stock_simple_returns < 0])
        momentum_score = winner_exposure - loser_exposure
        return portfolio_simple_return + 0.05 * momentum_score


class MarketTimingReward(RewardFunction):
    """Option 16: Market Timing Reward (reward for holding cash in down markets)"""
    
    def __init__(self):
        super().__init__("Market Timing Reward")
    
    def compute(self, portfolio_simple_return, cash_allocation, stock_simple_returns, **kwargs):
        market_return = np.mean(stock_simple_returns)
        if market_return < -0.01:
            cash_bonus = 0.03 * cash_allocation
        else:
            cash_bonus = 0.0
        return portfolio_simple_return + cash_bonus


class RegimeAdaptiveReward(RewardFunction):
    """Option 17: Regime-Adaptive Reward"""
    
    def __init__(self):
        super().__init__("Regime-Adaptive Reward")
    
    def compute(self, portfolio_simple_return, cash_allocation, allocation_changes, stock_simple_returns, **kwargs):
        market_return = np.mean(stock_simple_returns)
        market_volatility = np.std(stock_simple_returns)
        if market_volatility > 0.02:  # High volatility regime
            return portfolio_simple_return + 0.03 * cash_allocation
        else:  # Low volatility regime
            return portfolio_simple_return + 0.02 * allocation_changes


class TurnoverWeightedReturn(RewardFunction):
    """Option 18: Turnover-Weighted Return - Strong anti-buy-and-hold"""
    
    def __init__(self):
        super().__init__("Turnover-Weighted Return")
    
    def compute(self, portfolio_simple_return, allocation_changes, **kwargs):
        if allocation_changes < 0.05:
            activity_penalty = -0.01
        else:
            activity_penalty = 0.02 * allocation_changes
        return portfolio_simple_return + activity_penalty


class TacticalAllocationReward(RewardFunction):
    """Option 19: Tactical Allocation Reward"""
    
    def __init__(self):
        super().__init__("Tactical Allocation Reward")
    
    def compute(self, portfolio_simple_return, stock_allocations, stock_simple_returns, **kwargs):
        sorted_returns = np.argsort(stock_simple_returns)
        top_performers = sorted_returns[-3:]
        bottom_performers = sorted_returns[:3]
        tactical_score = (np.sum(stock_allocations[top_performers]) - 
                         np.sum(stock_allocations[bottom_performers]))
        return portfolio_simple_return + 0.03 * tactical_score


class DynamicRebalancingReward(RewardFunction):
    """Option 20: Dynamic Rebalancing Reward - Excellent for active strategy"""
    
    def __init__(self):
        super().__init__("Dynamic Rebalancing Reward")
    
    def compute(self, portfolio_simple_return, allocation_changes, stock_simple_returns, **kwargs):
        if allocation_changes > 0.01:
            equal_weight_return = np.mean(stock_simple_returns)
            if portfolio_simple_return > equal_weight_return:
                trade_quality = 0.05
            else:
                trade_quality = -0.02
        else:
            trade_quality = -0.01
        return portfolio_simple_return + trade_quality


class RelativeStrengthReward(RewardFunction):
    """Option 21: Relative Strength Reward"""
    
    def __init__(self):
        super().__init__("Relative Strength Reward")
    
    def compute(self, portfolio_simple_return, stock_allocations, stock_simple_returns, **kwargs):
        mean_return = np.mean(stock_simple_returns)
        relative_returns = stock_simple_returns - mean_return
        alignment = np.sum(stock_allocations * relative_returns)
        return portfolio_simple_return + 0.1 * alignment


class RiskOnRiskOffReward(RewardFunction):
    """Option 22: Risk-On/Risk-Off Reward"""
    
    def __init__(self):
        super().__init__("Risk-On Risk-Off Reward")
    
    def reset(self):
        """Reset state"""
        super().reset()
        self.state['recent_returns'] = []
    
    def compute(self, portfolio_simple_return, cash_allocation, stock_simple_returns, **kwargs):
        if 'recent_returns' not in self.state:
            self.state['recent_returns'] = []
            
        self.state['recent_returns'].append(np.mean(stock_simple_returns))
        if len(self.state['recent_returns']) > 10:
            self.state['recent_returns'].pop(0)
        
        trend = np.mean(self.state['recent_returns']) if len(self.state['recent_returns']) > 1 else 0
        if trend > 0:  # Risk-on environment
            return portfolio_simple_return + 0.02 * (1 - cash_allocation)
        else:  # Risk-off environment
            return portfolio_simple_return + 0.02 * cash_allocation


# ============================================================================
# REWARD REGISTRY
# ============================================================================

def get_all_reward_functions():
    """Return a list of all available reward function instances"""
    return [
        # Standard rewards
        RawPortfolioReturn(),
        LogReturn(),
        ReturnOverRollingVolatility(),
        OnlineSharpeRatioReward(),
        NormalizedProfit(),
        SortinoLikeReward(),
        ReturnMinusDrawdownPenalty(),
        ExcessReturnOverBenchmark(),
        RiskAdjustedByConcentration(),
        CalmarLikeReward(),
        ReturnPlusDiversificationBonus(),
        InformationRatioReward(),
        ConditionalValueAtRiskReward(),
        
        # Anti-buy-and-hold rewards
        TradingActivityBonus(),
        MomentumExploitationReward(),
        MarketTimingReward(),
        RegimeAdaptiveReward(),
        TurnoverWeightedReturn(),
        TacticalAllocationReward(),
        DynamicRebalancingReward(),
        RelativeStrengthReward(),
        RiskOnRiskOffReward(),
    ]


def get_reward_function_by_name(name):
    """Get a specific reward function by name"""
    all_rewards = get_all_reward_functions()
    for reward in all_rewards:
        if reward.name == name:
            return reward
    raise ValueError(f"Reward function '{name}' not found")
