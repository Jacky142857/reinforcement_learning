This project presents a comprehensive empirical study on the impact of reward function design on
the performance and stability of Deep Reinforcement Learning (DRL) agents in portfolio allocation.
Addressing common methodological limitations in existing literature—such as selection bias and
the lack of cash management options—this study utilizes the Dow Jones 30 constituent stocks as the
asset universe and explicitly incorporates a cash asset to allow for risk-off positioning.
We benchmark three prominent actor-critic algorithms—Proximal Policy Optimization (PPO), Soft
Actor-Critic (SAC), and Twin Delayed Deep Deterministic Policy Gradient (TD3)—across twenty-
two distinct reward formulations. These functions range from raw profit metrics to sophisticated
risk-adjusted and drawdown-penalized objectives. The agents were trained on data from January
2015 to December 2022 and evaluated on an out-of-sample period from January 2023 to December
2024 to systematically analyze how different reward structures influence trading behavior, risk
management, and convergence stability.
