"""
Actor-Critic RL Agent Module
Implements N-step Deep Reinforcement Learning for insulin control
Based on Wang, S., & Gu, W. (2024) and patent WO 2018/011766 A1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Tuple, Dict, List
import random


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])


class ReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.
    Samples experiences with higher TD-error more frequently.
    """
    
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Max buffer size
            alpha: Prioritization exponent [0, 1]. 0 = uniform sampling, 1 = full prioritization
            beta: Importance-sampling exponent for weight correction
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, experience: Experience, td_error: float = 1.0):
        """Store experience with TD-error-based priority."""
        # Priority = (|TD-error| + epsilon)^alpha
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray]:
        """
        Sample batch with probability proportional to priorities.
        
        Returns:
            (experiences, weights) where weights are importance-sampling corrections
        """
        
        if len(self.buffer) == 0:
            return [], np.array([])
        
        # Compute sampling probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probabilities,
            replace=False
        )
        
        # Importance-sampling weights (correct for bias from non-uniform sampling)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights
    
    def __len__(self) -> int:
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """
    Actor: Proposes continuous insulin dose actions.
    Input: normalized state vector
    Output: mean and log-std of Gaussian policy
    """
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Log standard deviation (learnable, not output)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (mean_action, log_std) for reparameterization trick
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mean = self.fc3(x)
        
        # Clamp mean to [0, 10] units/min (insulin bounds)
        mean = torch.clamp(mean, 0, 10) * self.tanh(mean)
        
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy (with reparameterization trick)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            return mean
        
        # Reparameterization trick
        eps = torch.randn_like(std)
        action = mean + eps * std
        
        # Clamp to valid insulin range
        action = torch.clamp(action, 0, 10)
        
        return action


class CriticNetwork(nn.Module):
    """
    Critic: Estimates value of state (expected cumulative reward).
    Input: state + action
    Output: Q-value scalar
    """
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        # Concatenate state and action
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output: Q-value
        
        self.relu = nn.ReLU()
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ActorCriticAgent:
    """
    Deep Reinforcement Learning agent for glucose control.
    
    Key features:
    - N-step returns to handle insulin action delay (Wang et al. 2024)
    - Prioritized Experience Replay (PER) for efficient learning
    - Actor-Critic architecture with separate policy and value networks
    - Clipped objective to prevent mode collapse
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        n_steps: int = 16,  # 30-45 min insulin delay / 5 min per step
        entropy_coeff: float = 0.01,
        device: str = 'cpu',
    ):
        """
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action (1 for insulin dose)
            learning_rate: Adam optimizer learning rate
            discount_factor: Gamma in RL
            n_steps: Number of steps for multi-step return (addresses insulin delay)
            entropy_coeff: Coefficient for entropy regularization
            device: 'cpu' or 'cuda'
        """
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.device = device
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        
        # Copy target network
        self._soft_update_target(tau=1.0)
        
        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100_000)
        self.n_step_buffer = deque(maxlen=n_steps)
        
        # Training statistics
        self.total_updates = 0
        self.critic_loss_history = []
        self.actor_loss_history = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """
        Select insulin dose action given current state.
        
        Args:
            state: Normalized state vector
            deterministic: If True, return mean (no exploration)
            
        Returns:
            Insulin dose (units/min)
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor.sample_action(state_tensor, deterministic=deterministic)
        
        return float(action.cpu().numpy()[0])
    
    def store_experience(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Dict = None
    ):
        """
        Store experience in N-step buffer, then to replay buffer when N-step return is ready.
        """
        
        if info is None:
            info = {}
        
        # Store in n-step buffer
        exp = Experience(state, action, reward, next_state, done, info)
        self.n_step_buffer.append(exp)
        
        # When we have n steps, compute n-step return and store in replay buffer
        if len(self.n_step_buffer) == self.n_steps:
            self._process_n_step_experience()
    
    def _process_n_step_experience(self):
        """
        Compute n-step return and add to replay buffer.
        
        R_t^(n) = Σ(γ^k * r_{t+k}) + γ^n * Q(s_{t+n}, a_{t+n})
        """
        
        # Get first and last experiences
        first_exp = self.n_step_buffer[0]
        last_exp = self.n_step_buffer[-1]
        
        # Compute n-step discounted return
        n_step_return = 0.0
        for i, exp in enumerate(self.n_step_buffer):
            n_step_return += (self.discount_factor ** i) * exp.reward
        
        # Bootstrap from n-step state (using target network)
        if not last_exp.done:
            next_state_tensor = torch.from_numpy(last_exp.next_state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Sample action from current policy
                next_action = self.actor.sample_action(next_state_tensor)
                # Estimate value using target critic
                bootstrap_value = self.critic_target(next_state_tensor, next_action)
            n_step_return += (self.discount_factor ** self.n_steps) * float(bootstrap_value.cpu().numpy()[0])
        
        # Compute TD error for prioritization
        state_tensor = torch.from_numpy(first_exp.state).float().unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([[first_exp.action]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            current_value = self.critic(state_tensor, action_tensor)
            td_error = n_step_return - float(current_value.cpu().numpy()[0])
        
        # Create experience with n-step return
        n_step_exp = Experience(
            first_exp.state,
            first_exp.action,
            n_step_return,
            last_exp.next_state,
            last_exp.done,
            {**first_exp.info, 'n_step': self.n_steps}
        )
        
        # Add to replay buffer with TD-error priority
        self.replay_buffer.push(n_step_exp, td_error=abs(td_error))
    
    def update(self, batch_size: int = 32):
        """
        Update actor and critic networks using mini-batch from replay buffer.
        """
        
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch with prioritized replay
        experiences, weights = self.replay_buffer.sample(batch_size)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        # Convert to tensors
        states = torch.stack([
            torch.from_numpy(exp.state).float() for exp in experiences
        ]).to(self.device)
        
        actions = torch.tensor([[exp.action] for exp in experiences], dtype=torch.float32).to(self.device)
        n_step_returns = torch.tensor([[exp.reward] for exp in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([
            torch.from_numpy(exp.next_state).float() for exp in experiences
        ]).to(self.device)
        dones = torch.tensor([[0.0 if not exp.done else 1.0] for exp in experiences], dtype=torch.float32).to(self.device)
        
        # === Update Critic ===
        with torch.no_grad():
            next_actions = self.actor.sample_action(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = n_step_returns + self.discount_factor ** self.n_steps * (1 - dones) * target_q
        
        q_values = self.critic(states, actions)
        critic_loss = (weights * (q_values - target) ** 2).mean()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        
        # === Update Actor ===
        actions_new = self.actor.sample_action(states)
        actor_loss = -self.critic(states, actions_new).mean()
        
        # Entropy regularization (encourage exploration)
        mean, log_std = self.actor.forward(states)
        std = log_std.exp()
        entropy = (log_std + 0.5 * np.log(2 * np.pi * np.e)).mean()
        
        actor_loss = actor_loss - self.entropy_coeff * entropy
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()
        
        # === Soft update target network ===
        self._soft_update_target(tau=0.005)
        
        # Track losses
        self.critic_loss_history.append(float(critic_loss.detach().cpu().numpy()))
        self.actor_loss_history.append(float(actor_loss.detach().cpu().numpy()))
        self.total_updates += 1
    
    def _soft_update_target(self, tau: float = 0.005):
        """Soft update target network: θ_target ← τ*θ + (1-τ)*θ_target"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save actor and critic networks."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load actor and critic networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    
    print("=== Actor-Critic Agent Demo ===\n")
    
    # Initialize agent
    state_dim = 22  # From state_encoder.get_state_dim()
    agent = ActorCriticAgent(
        state_dim=state_dim,
        learning_rate=1e-4,
        n_steps=16,
    )
    
    print(f"Agent initialized:")
    print(f"  State dim: {state_dim}")
    print(f"  N-step returns: {agent.n_steps}")
    print(f"  Actor: {agent.actor}")
    print(f"  Critic: {agent.critic}\n")
    
    # Simulate one episode
    print("Simulating one episode (10 steps)...")
    
    state = np.random.randn(state_dim).astype(np.float32)
    
    for step in range(10):
        # Agent selects action
        action = agent.select_action(state)
        
        # Simulate environment
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = -0.5 * (step ** 2)  # Example reward
        done = (step == 9)
        
        # Store experience
        agent.store_experience(state, action, reward, next_state, done)
        
        print(f"  Step {step}: action={action:.3f}, reward={reward:.3f}")
        
        state = next_state
        
        # Update agent every 2 steps
        if step % 2 == 0:
            agent.update(batch_size=4)
    
    print(f"\nTraining complete. Total updates: {agent.total_updates}")
    print(f"Avg critic loss: {np.mean(agent.critic_loss_history):.4f}")
    print(f"Avg actor loss: {np.mean(agent.actor_loss_history):.4f}")