# ============================================================================
# NEURAL NETWORK DEFINITIONS
# ============================================================================

"""
here i define first the actor network that takes as input the resource unit state
 [allocation status,Ru_QI,queue_pressure which are the number of packets waiting for each station
 here i propose that are the same for all stations,deadline_pressure 
 which is the waiting time pressure for packet transmission
  the same proposition as the previous one]
 and return the probability of allocating that resource unit for each station.
"""
class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

'''
then i define the critic network that takes as input the global state
 of the OFDMA channel which is the concatenation of all resource units states 
 and return the value of that state which is the expected cumulative discounted reward from that state.
'''
class CriticNetwork(nn.Module):
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


