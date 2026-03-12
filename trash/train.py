"""
Main Training Loop
Integrates Environment, RL Agent, Safety Layer, and Reward Shaping
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from datetime import datetime
import json
from trash.simglucose_env import SimGlucoseEnvWrapper
from state_encoder import StateEncoder
from safety_layer import SafetyLayer
from reward_shaper import RewardShaper
from actor_critic_agent import ActorCriticAgent
    
class GlucoseControlEnvironment:
    """
    Mock environment simulating diabetes + glucose dynamics.
    In production, replace with SimGlucose wrapper.
    """
    
    def __init__(self, patient_params: Dict = None):
        """
        Args:
            patient_params: Patient-specific parameters (CGM noise, insulin sensitivity, etc.)
        """
        
        if patient_params is None:
            patient_params = {}
        
        self.cgm_noise_std = patient_params.get('cgm_noise_std', 5.0)
        self.insulin_sensitivity = patient_params.get('insulin_sensitivity', 1.0)  # mg/dL per unit insulin
        self.meal_impact = patient_params.get('meal_impact', 40.0)  # mg/dL per carb
        
        # State
        self.glucose = 130.0
        self.active_insulin = 0.0
        self.time_step = 0
        self.max_steps = 288  # 24 hours at 5-min intervals
        
    def reset(self) -> float:
        """Reset for new episode."""
        self.glucose = 130.0 + np.random.randn() * 10
        self.active_insulin = 0.0
        self.time_step = 0
        return self.glucose
    
    def step(self, insulin_dose: float, meal_carbs: float = 0.0) -> Tuple[float, bool]:
        """
        Step environment given insulin action.
        
        Args:
            insulin_dose: Insulin delivered (units/min)
            meal_carbs: Carbohydrates consumed (grams) - for routine learning
            
        Returns:
            (glucose, done) tuple
        """
        
        # Update active insulin (exponential decay, 4-hour half-life)
        self.active_insulin = self.active_insulin * 0.97 + insulin_dose
        
        # Glucose dynamics (simplified)
        # dG/dt = -k * I + meal_effect - basal_usage
        insulin_effect = -self.active_insulin * self.insulin_sensitivity
        meal_effect = meal_carbs * self.meal_impact / 60  # Distributed over hour
        basal_usage = -1.0  # Natural glucose consumption
        
        glucose_change = insulin_effect + meal_effect + basal_usage
        
        # Add realistic noise
        noise = np.random.randn() * self.cgm_noise_std
        
        self.glucose = max(10, self.glucose + glucose_change + noise)  # Min 10 (safety)
        
        self.time_step += 1
        done = self.time_step >= self.max_steps
        
        return self.glucose, done


class TrainingLoop:
    """
    Main training loop coordinating all components.
    """
    
    def __init__(
        self,
        agent,
        safety_layer,
        reward_shaper,
        state_encoder,
        env: GlucoseControlEnvironment,
        log_dir: str = './logs',
    ):
        """
        Args:
            agent: ActorCriticAgent
            safety_layer: SafetyLayer
            reward_shaper: RewardShaper
            state_encoder: StateEncoder
            env: GlucoseControlEnvironment
            log_dir: Directory for logging
        """
        
        self.agent = agent
        self.safety_layer = safety_layer
        self.reward_shaper = reward_shaper
        self.state_encoder = state_encoder
        self.env = env
        self.log_dir = log_dir
        
        # Training logs
        self.episode_rewards = []
        self.episode_tir = []
        self.episode_tbr = []
        self.episode_metrics = []
    
    def train_episode(self, deterministic: bool = False) -> Dict:
        """
        Run one training episode (24 hours of glucose control).
        
        Returns:
            Episode summary dictionary
        """
        
        # Reset environment and agent
        glucose = self.env.reset()
        self.safety_layer.reset()
        self.state_encoder.reset()
        
        episode_data = {
            'glucose': [],
            'insulin': [],
            'rewards': [],
            'actions': [],
        }
        
        total_reward = 0.0
        previous_action = 0.0
        
        # 24-hour episode (288 steps at 5-min intervals)
        for step in range(self.env.max_steps):
            
            # Current time of day
            time_hours = (step * 5) / 60  # 0-23.99
            
            # Encode state
            activity_level = 0.1 * np.sin(step / 50)  # Simulated activity
            routine_context = {'meal_likely': 0.0}  # Placeholder
            
            state = self.state_encoder.encode(
                glucose=glucose,
                time_hours=time_hours,
                active_insulin=self.env.active_insulin,
                activity_level=max(0, activity_level),
                routine_context=routine_context,
            )
            
            # Flatten state
            state_vec = self.state_encoder.flatten_state(state)
            
            # RL agent proposes action
            rl_proposed = self.agent.select_action(state_vec, deterministic=deterministic)
            
            # Safety layer validates
            safe_insulin, safety_msg, is_safe = self.safety_layer.evaluate_action(
                rl_proposed_insulin=rl_proposed,
                glucose=glucose,
                active_insulin=self.env.active_insulin,
                previous_action=previous_action,
                use_pid_override=True,
            )
            
            # Environment step
            next_glucose, done = self.env.step(safe_insulin)
            
            # Reward computation
            reward_dict = self.reward_shaper.compute_reward(
                glucose=next_glucose,
                previous_glucose=glucose,
                insulin_dose=safe_insulin,
                previous_insulin=previous_action,
                active_insulin=self.env.active_insulin,
            )
            reward = reward_dict['total']
            
            # Store next state
            next_state = self.state_encoder.encode(
                glucose=next_glucose,
                time_hours=time_hours + (5/60),
                active_insulin=self.env.active_insulin,
                activity_level=max(0, activity_level),
                routine_context=routine_context,
            )
            next_state_vec = self.state_encoder.flatten_state(next_state)
            
            # Store in agent's replay buffer
            self.agent.store_experience(
                state=state_vec,
                action=safe_insulin,
                reward=reward,
                next_state=next_state_vec,
                done=done,
                info={'step': step, 'glucose': glucose, 'is_safe': is_safe},
            )
            
            # Update agent
            if step % 2 == 0 and len(self.agent.replay_buffer) > 32:
                self.agent.update(batch_size=32)
            
            # Logging
            episode_data['glucose'].append(float(next_glucose))
            episode_data['insulin'].append(float(safe_insulin))
            episode_data['rewards'].append(float(reward))
            episode_data['actions'].append(float(rl_proposed))
            
            total_reward += reward
            
            # Update for next iteration
            glucose = next_glucose
            previous_action = safe_insulin
            
            if done:
                break
        
        # Compute episode metrics
        glucose_array = np.array(episode_data['glucose'])
        tir = 100 * np.sum((glucose_array >= 70) & (glucose_array <= 180)) / len(glucose_array)
        tbr = 100 * np.sum(glucose_array < 70) / len(glucose_array)
        
        summary = {
            'episode_reward': total_reward,
            'mean_glucose': np.mean(glucose_array),
            'std_glucose': np.std(glucose_array),
            'tir': tir,
            'tbr': tbr,
            'steps': len(episode_data['glucose']),
            'data': episode_data,
        }
        
        # Track metrics
        self.episode_rewards.append(total_reward)
        self.episode_tir.append(tir)
        self.episode_tbr.append(tbr)
        self.episode_metrics.append(summary)
        
        return summary
    
    def run_training(
        self,
        num_episodes: int = 100,
        eval_interval: int = 10,
        save_interval: int = 50,
    ):
        """
        Run full training loop.
        
        Args:
            num_episodes: Number of episodes to train
            eval_interval: Evaluate every N episodes
            save_interval: Save model every N episodes
        """
        
        print(f"Starting training: {num_episodes} episodes")
        print(f"{'Episode':<10} {'Reward':<12} {'TIR %':<10} {'TBR %':<10} {'Avg Loss':<12}")
        print("-" * 60)
        
        for episode in range(num_episodes):
            
            # Training episode
            summary = self.train_episode(deterministic=False)
            
            # Logging every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_loss = np.mean(self.agent.critic_loss_history[-100:]) if self.agent.critic_loss_history else 0
                print(
                    f"{episode + 1:<10} "
                    f"{summary['episode_reward']:<12.2f} "
                    f"{summary['tir']:<10.1f} "
                    f"{summary['tbr']:<10.1f} "
                    f"{avg_loss:<12.4f}"
                )
            
            # Evaluation
            if (episode + 1) % eval_interval == 0:
                self._evaluate(num_eval=3)
            
            # Checkpoint
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1)
        
        print("\nTraining complete!")
        self._save_results()
    
    def _evaluate(self, num_eval: int = 3):
        """Run evaluation episodes with deterministic policy."""
        eval_tirs = []
        
        for _ in range(num_eval):
            summary = self.train_episode(deterministic=True)
            eval_tirs.append(summary['tir'])
        
        mean_tir = np.mean(eval_tirs)
        print(f"  [EVAL] Mean TIR: {mean_tir:.1f}%")
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        filepath = f"{self.log_dir}/agent_episode_{episode}.pt"
        self.agent.save(filepath)
        print(f"  [SAVE] Checkpoint saved: {filepath}")
    
    def _save_results(self):
        """Save training results to JSON."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'episode_tir': self.episode_tir,
            'episode_tbr': self.episode_tbr,
            'final_tir': self.episode_tir[-1] if self.episode_tir else 0,
        }
        
        filepath = f"{self.log_dir}/training_results.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Example: Run training loop."""
    
    print("="*70)
    print("Routine-Aware RL Artificial Pancreas - Training")
    print("="*70 + "\n")
    
    # Import components (you'll have these from separate files)
    # For now, mock imports
    # Initialize components
    print("Initializing components...")
    
    state_encoder = StateEncoder(glucose_history_size=11, target_glucose=130)
    state_dim = state_encoder.get_state_dim()
    
    agent = ActorCriticAgent(
        state_dim=state_dim,
        learning_rate=1e-4,
        n_steps=16,
        device='cpu',  # Change to 'cuda' if GPU available
    )
    
    safety_layer = SafetyLayer(
        max_insulin_per_minute=10.0,
        target_glucose=130.0,
    )
    
    reward_shaper = RewardShaper(
        target_low=70.0,
        target_high=180.0,
    )
    
    # env = GlucoseControlEnvironment(
    #     patient_params={
    #         'cgm_noise_std': 5.0,
    #         'insulin_sensitivity': 1.0,
    #     }
    # )

    env = SimGlucoseEnvWrapper()


    # Create training loop
    trainer = TrainingLoop(
        agent=agent,
        safety_layer=safety_layer,
        reward_shaper=reward_shaper,
        state_encoder=state_encoder,
        env=env,
        log_dir='./logs',
    )
    
    # Run training
    trainer.run_training(
        num_episodes=100,  # Start with 50, scale to 500+
        eval_interval=10,
        save_interval=25,
    )


if __name__ == "__main__":
    main()