
# ============================================================================
# TRAJECTORY DATA STRUCTURE
# ============================================================================

class Trajectory(NamedTuple):
    """
    Single trajectory tuple collected during rollout.
    
    (state, action, log_prob, reward, value, rtg)
    """
    state: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    rtg: float = 0.0


# ============================================================================
# REWARDS-TO-GO AND ADVANTAGE CALCULATION
# ============================================================================

def compute_rewards_to_go(
    rewards: np.ndarray,
    gamma: float,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute Rewards-To-Go (RTG) for entire trajectory.
    
    RTG_t = Σ_{k=0}^{T-t-1} γ^k * r_{t+k}
        = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...
    """
    T = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    rtg[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        rtg[t] = rewards[t] + gamma * rtg[t + 1]
    
    if normalize:
        rtg = (rtg - rtg.mean()) / (rtg.std() + 1e-8)
    
    return rtg


def compute_advantages(
    rtg: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """
    Compute Advantage Estimates: A(s,a) = Q(s,a) - V(s)
    """
    advantages = rtg - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute done-aware GAE advantages and returns."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    next_adv = 0.0
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        next_adv = delta + gamma * lam * mask * next_adv
        advantages[t] = next_adv
        returns[t] = advantages[t] + values[t]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns
