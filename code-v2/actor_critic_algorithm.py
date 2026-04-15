'''
here we define the agent configuration the we set the agent where we combine the previous result of enviroment and the networks and the optimizers and the training history which
 consists of metrics that we need to use 
 after that we collect the trajectories by interacting with the environment using the current policy
 
  and we store the transitions in the trajectory data structure 
  then we compute the rewards-to-go and the advantages 
  and we update the critic network by minimizing the MSE loss between the predicted value and the RTG and we update the actor network by maximizing the policy gradient loss with an entropy bonus for exploration 
  then we record the metrics and print them every 10 episodes
'''

# ============================================================================
# ACTOR-CRITIC ALGORITHM: MAIN TRAINING LOOP
# ============================================================================



'''
this is the configuration of the training process
'''
@dataclass
class ActorCriticConfig:
    """Configuration for Actor-Critic training."""
    n_stations: int = 5
    num_episodes: int = 100
    rollout_length: int = 256
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 1e-4
    entropy_coef: float = 0.02
    entropy_coef_end: float = 0.002
    entropy_decay_episodes: int = 100
    gae_lambda: float = 0.95
    critic_updates: int = 4
    actor_updates: int = 2
    max_grad_norm: float = 1.0
    normalize_returns: bool = True
    device: str = "cpu"
    seed: int = 42


class ActorCriticAgent:
    
    def __init__(self, config: ActorCriticConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        
        # Initialize environment first to get actual number of RUs
        self.env = OFDMAEnvironment(n_stations=config.n_stations)
        self.n_rus = self.env.n_rus
        self.n_stations = config.n_stations
        
        # Get actual state dimension from environment
        dummy_state = self.env.reset()
        self.state_dim = len(dummy_state)  # Flattened state size
        self.action_dim = self.n_stations + 1  # 0: no allocation, 1..n_stations: allocate to station
        self.actor_output_dim = self.n_rus * self.action_dim
        
        # Initialize networks
        self.actor = ActorNetwork(self.state_dim, self.actor_output_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.current_entropy_coef = config.entropy_coef
        
        # Metrics
        self.training_history = {
            "episode_rewards": [],
            "critic_loss": [],
            "actor_loss": [],
            "entropy": [],
            "max_action_share": [],
            "throughput": [],
            "collision_rate": [],
        }
    
    def collect_trajectories(self) -> tuple:


        state = self.env.reset()
        
        trajectories = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }
        action_hist = np.zeros(self.action_dim, dtype=np.int64)
        step_throughputs = []
        step_collision_rates = []
        
        episode_reward = 0.0
        
        for step in range(self.cfg.rollout_length):
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            # Get value estimate from critic
            with torch.no_grad():
                value = self.critic(state_tensor).item()
            
            # Sample one action per RU from reshaped logits: (n_rus, action_dim)
            with torch.no_grad():
                logits = self.actor(state_tensor).view(self.n_rus, self.action_dim)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_prob = dist.log_prob(actions).sum().item()

            actions_np = actions.cpu().numpy().astype(np.int32)
            action_hist += np.bincount(actions_np, minlength=self.action_dim)
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(actions_np)
            step_throughputs.append(float(info.get("throughput", 0.0)))
            step_collision_rates.append(float(info.get("collision_rate", 0.0)))
            
            # Store trajectory
            trajectories["states"].append(state.copy())
            trajectories["actions"].append(actions_np.copy())
            trajectories["log_probs"].append(log_prob)
            trajectories["rewards"].append(reward)
            trajectories["values"].append(value)
            trajectories["dones"].append(done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                state = self.env.reset()

        total_actions = int(action_hist.sum())
        max_action_share = float(action_hist.max() / total_actions) if total_actions > 0 else 0.0
        trajectories["max_action_share"] = max_action_share
        trajectories["throughput"] = float(np.mean(step_throughputs)) if step_throughputs else 0.0
        trajectories["collision_rate"] = (
            float(np.mean(step_collision_rates)) if step_collision_rates else 0.0
        )
        
        return trajectories, episode_reward
    

    def update_networks(self, trajectories: dict) -> tuple:
        """
        PHASES 2-4: POST-PROCESSING AND NETWORK UPDATES
        """
        
        # Phase 2: Compute RTG and Advantages
        rewards = np.array(trajectories["rewards"], dtype=np.float32)
        values = np.array(trajectories["values"], dtype=np.float32)
        
        dones = np.array(trajectories["dones"], dtype=np.float32)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
        )
        if self.cfg.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.tensor(
            np.array(trajectories["states"]),
            dtype=torch.float32,
            device=self.device
        )
        actions_tensor = torch.tensor(
            np.array(trajectories["actions"]),
            dtype=torch.int64,
            device=self.device
        )
        advantages_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device
        )
        returns_tensor = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device
        )
        
        # Phase 3: Critic Update
        critic_losses = []
        for _ in range(self.cfg.critic_updates):
            value_pred = self.critic(states_tensor).squeeze(-1)
            loss = torch.nn.functional.smooth_l1_loss(value_pred, returns_tensor)

            self.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_optimizer.step()

            critic_losses.append(loss.item())
        
        avg_critic_loss = np.mean(critic_losses)
        
        # Phase 4: Actor Update
        actor_losses = []
        entropies = []
        for _ in range(self.cfg.actor_updates):
            logits = self.actor(states_tensor).view(-1, self.n_rus, self.action_dim)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions_tensor).sum(dim=1)

            loss_policy = -(log_probs * advantages_tensor).mean()
            entropy = dist.entropy().mean(dim=1).mean()
            loss_actor = loss_policy - self.current_entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_optimizer.step()

            actor_losses.append(loss_actor.item())
            entropies.append(entropy.item())

        return avg_critic_loss, float(np.mean(actor_losses)), float(np.mean(entropies))
    
    def train(self):
        """Main training loop."""
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        
        for episode in range(self.cfg.num_episodes):
            decay_ratio = min(1.0, episode / max(1, self.cfg.entropy_decay_episodes - 1))
            self.current_entropy_coef = (
                self.cfg.entropy_coef
                + decay_ratio * (self.cfg.entropy_coef_end - self.cfg.entropy_coef)
            )

            # Phase 1: Collect trajectories
            trajectories, ep_reward = self.collect_trajectories()
            
            # Phases 2-4: Update networks
            critic_loss, actor_loss, entropy = self.update_networks(trajectories)
            
            # Record metrics
            self.training_history["episode_rewards"].append(ep_reward)
            self.training_history["critic_loss"].append(critic_loss)
            self.training_history["actor_loss"].append(actor_loss)
            self.training_history["entropy"].append(entropy)
            self.training_history["max_action_share"].append(trajectories["max_action_share"])
            self.training_history["throughput"].append(trajectories["throughput"])
            self.training_history["collision_rate"].append(trajectories["collision_rate"])
            
            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1:3d}/{self.cfg.num_episodes} | "
                    f"Reward: {ep_reward:7.3f} | "
                    f"Critic Loss: {critic_loss:.4f} | "
                    f"Actor Loss: {actor_loss:.4f} | "
                    f"Entropy: {entropy:.4f} | "
                    f"Max Action Share: {trajectories['max_action_share']:.3f} | "
                    f"Throughput: {trajectories['throughput']:.3f} | "
                    f"Collision Rate: {trajectories['collision_rate']:.3f}"
                )
        
        return self.training_history


def train_actor_critic(cfg: ActorCriticConfig | None = None) -> dict:
    """Train actor-critic agent on OFDMA environment."""
    if cfg is None:
        cfg = ActorCriticConfig()
    
    agent = ActorCriticAgent(cfg)
    history = agent.train()
    
    return {
        "history": history,
        "agent": agent,
        "config": cfg,
    }


if __name__ == "__main__":
    cfg = ActorCriticConfig()
    result = train_actor_critic(cfg)
    print("\n✓ Actor-Critic training completed!")

