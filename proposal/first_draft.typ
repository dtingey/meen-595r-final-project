= ME EN 595R Final Project Proposal
Damon Tingey

My project studies decentralized coordination for a multi-agent search-and-control task. I want to understand how a team of agents can cover an environment efficiently while avoiding redundant behavior and poor coordination when no centralized planner is available. I will formulate this as a finite potential game in which each agent’s local utility is tied to a shared team objective.

I will implement binary log-linear learning (BLLL) as the primary solution method and compare it with deterministic best-response learning from Chapter 24. Both methods will be tested on the same scenarios while varying problem size and team size to evaluate robustness and scalability. I will measure convergence speed, final objective value, policy stability, and computational cost.

This approach is a good fit because potential games give a principled framework for aligning local decisions with global performance. BLLL is well suited to decentralized settings because stochastic updates can help agents avoid poor local equilibria, while best response provides a strong baseline for comparison. The comparison should clarify when stochastic learning offers meaningful performance gains for cooperative multi-agent control.