# Routine-Aware RL Artificial Pancreas - Python Implementation

A Deep Reinforcement Learning system for autonomous insulin delivery in Type 1 Diabetes management.

## Quick Start

### 1. Install Dependencies

```bash
# Core ML libraries
pip install torch==2.1.0
pip install numpy pandas

# Optional: SimGlucose for realistic simulation
pip install simglucose==0.2.3

# Optional: Visualization & monitoring
pip install matplotlib tensorboard scipy
```

### 2. Run Example Training

```bash
python train.py
```

This will:
- Initialize the RL agent, safety layer, and simulator
- Run 50 training episodes (each simulating 24 hours)
- Track Time-in-Range (TIR) and other metrics
- Save checkpoints and results

### 3. Examine Generated Logs

```bash
# View training results
cat logs/training_results.json

# Load trained model
agent.load('logs/agent_episode_50.pt')
```

---

## Project Structure

```
├── state_encoder.py          # Normalize sensor data → state vector
├── safety_layer.py           # PID fallback + constraint validation
├── reward_shaper.py          # Define glucose control objective
├── actor_critic_agent.py     # RL agent with N-step returns
├── train.py                  # Main training loop
├── README.md                 # This file
└── logs/                     # Training outputs (created at runtime)
```

---

## Architecture Overview

### System Diagram

```
    Sensors (CGM + Wearables)
            ↓
        State Encoder        ← Glucose, time-of-day, activity
            ↓
        RL Agent (Actor)     ← Proposes insulin dose
            ↓
        Safety Layer         ← Validates against constraints
            ↓
        Insulin Pump         ← Delivers safe dose
            ↓
        Glucose Response     ← Measured by CGM
            ↓
        Reward Function      ← Guides learning toward TIR
```

### Core Components

#### 1. **State Encoder** (`state_encoder.py`)
Converts raw sensor signals into normalized RL-ready state vector.

**Inputs:**
- Glucose: Current + 11-step history (mg/dL)
- Time: Circular encoding (sin/cos) to handle midnight wraparound
- Activity: Accelerometer + GSR (0-1 normalized)
- Active Insulin: Residual insulin from previous doses (units)
- Routine Context: Probability of upcoming meals (from pattern learner)

**Output:** Normalized state vector (dimension 22)

**Key Methods:**
```python
encoder = StateEncoder(glucose_history_size=11, target_glucose=130)
state = encoder.encode(glucose=150, time_hours=12.5, active_insulin=3.5)
state_vec = encoder.flatten_state(state)  # Convert to 1D array
```

#### 2. **Safety Layer** (`safety_layer.py`)
Hard constraints + PID fallback (patent US 2022/0088304 A1).

**Constraints:**
- Max insulin: 10 units/min
- Max rate of change: 5 units/min
- Hypoglycemia protection: Zero insulin if BG < 70
- Insulin stacking prevention (Insulin Feedback buffer)

**Fallback:**
If RL action violates constraints → Use PID controller

**Key Methods:**
```python
safety = SafetyLayer(max_insulin_per_minute=10.0)
approved_dose, reason, is_safe = safety.evaluate_action(
    rl_proposed_insulin=8.0,
    glucose=150.0,
    active_insulin=2.0,
    previous_action=1.5,
)
```

#### 3. **Reward Shaper** (`reward_shaper.py`)
Multi-objective reward function balancing control and safety.

**Components:**
- Time-in-Range (70-180 mg/dL): +1.0 per step
- Tight Control (100-160 mg/dL): +0.5 bonus
- Hypoglycemia (BG < 70): -10.0 (scaled by severity)
- Severe Hypo (BG < 54): -100.0 (catastrophic)
- Hyperglycemia (BG > 180): -0.5 (mild)
- Insulin Smoothness: -0.1 × |ΔInsulin|
- Efficiency: +0.05 if TIR with minimal insulin

**Key Methods:**
```python
shaper = RewardShaper(target_low=70, target_high=180)
rewards = shaper.compute_reward(
    glucose=150.0,
    previous_glucose=148.0,
    insulin_dose=2.0,
    previous_insulin=1.8,
)
print(rewards['total'])  # Total reward signal
```

#### 4. **Actor-Critic Agent** (`actor_critic_agent.py`)
Deep RL agent addressing insulin action delay (Wang et al. 2024).

**Key Innovation: N-Step Returns**

Standard RL fails because insulin has 30-45 min delay. Standard Q-learning assigns reward only 1 step later, but insulin effect is minimal at t+1.

Solution: N-step return combines rewards over 16 steps (~45 min):

```
R_t^(n) = Σ(γ^k * r_{t+k}) + γ^n * max_a' Q(s_{t+n}, a')
         k=0 to n-1
```

**Architecture:**
- **Actor:** Proposes insulin dose (Gaussian policy)
- **Critic:** Estimates value (expected cumulative reward)
- **Replay Buffer:** Prioritized Experience Replay (PER) — samples high TD-error transitions more often

**Key Methods:**
```python
agent = ActorCriticAgent(state_dim=22, n_steps=16)

# Training
for episode in range(num_episodes):
    state = env.reset()
    for step in range(288):  # 24 hours
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.update(batch_size=32)
        state = next_state

# Inference
agent.eval()
action = agent.select_action(state, deterministic=True)
```

---

## Training Details

### Hyperparameters (Tunable)

```python
# RL Config
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99        # γ: importance of future rewards
N_STEPS = 16                   # Multi-step lookahead (30-45 min)
ENTROPY_COEFF = 0.01          # Exploration bonus
BATCH_SIZE = 32
BUFFER_CAPACITY = 100_000

# Environment
SIMULATION_HOURS = 24          # Seconds per episode
TIMESTEP_MIN = 5               # CGM reading interval (min)
NUM_EPISODES = 500             # Total training episodes
```

### Training Loop

1. **Reset environment** → Initialize glucose at 130 mg/dL
2. **For each 5-min step (288 steps/day):**
   - Encode state from sensors
   - RL agent proposes insulin dose
   - Safety layer approves/modifies
   - Environment simulates glucose response
   - Compute reward
   - Store in replay buffer
   - Update networks from mini-batch
3. **Evaluate** every 10 episodes with deterministic policy
4. **Save checkpoint** every 50 episodes

### Metrics Tracked

During training:
- **TIR (70-180 mg/dL):** Target is >70%
- **TBR (<70 mg/dL):** Target is <5%
- **Mean Glucose:** Target ~130 mg/dL
- **Hypo Episodes:** Number of separate low-glucose events
- **Total Insulin:** Daily insulin requirement
- **Cumulative Reward:** Training progress indicator

### Expected Results (from literature)

After 500 episodes on SimGlucose:
- TIR: ~85% (vs 70% baseline PID)
- TBR: ~2-3% (vs 5% baseline)
- Mean HbA1c: ~7.0% (vs 7.5% baseline)

---

## How to Extend: Routine Learner

Current implementation has placeholder for routine learning. To add it:

### 1. Pattern Recognition Module

```python
# routine_learner.py
class RoutineLearner:
    """Detects recurring daily patterns (meal times, exercise)."""
    
    def __init__(self, history_days=14):
        self.glucose_history = deque(maxlen=288*history_days)  # 14 days
        self.meal_times = []  # Detected meal times
        self.exercise_times = []
    
    def update(self, glucose, active_insulin, time_hours):
        """Update pattern recognition."""
        self.glucose_history.append((glucose, active_insulin, time_hours))
    
    def get_context(self, time_hours):
        """Return probability of upcoming meal/exercise."""
        # Use Fourier analysis or clustering to detect patterns
        # Return {'meal_likely': 0.8, 'exercise_likely': 0.1}
        pass
```

### 2. Integrate into Training

```python
# In train.py:
routine_learner = RoutineLearner(history_days=14)

for step in range(288):
    routine_context = routine_learner.get_context(time_hours)
    state = state_encoder.encode(..., routine_context=routine_context)
    
    # ... agent step ...
    
    routine_learner.update(glucose, active_insulin, time_hours)
```

---

## Testing Checklist

- [ ] Environment runs 24 hours without errors
- [ ] State encoder produces normalized values [-1, 1]
- [ ] Safety layer blocks insulin > 10 units/min
- [ ] Agent learns: TIR improves over 100 episodes
- [ ] N-step reduces TD error vs single-step
- [ ] Hypoglycemia triggers insulin halt
- [ ] PID fallback activates for unsafe RL actions
- [ ] Model saves/loads correctly

---

## Validation Against Literature

### Wang, S., & Gu, W. (2024)

**Claim:** N-step Q-learning achieves 85.62% TIR

**Our validation:**
```python
# Train agent with n_steps=16
agent = ActorCriticAgent(state_dim=22, n_steps=16)
# ...
# Evaluate TIR after convergence
# Expected: ~85% TIR
```

### Emerson, H., et al. (2023)

**Claim:** TD3-BC (offline RL) achieves 65.3% TIR safely

**Our validation:**
```python
# Implement TD3-BC variant
# Pre-train on historical data (no online exploration)
# Expected: 65% TIR with 0% failure rate
```

---

## Real-World Deployment Considerations

### Hardware Requirements

- **CPU:** ARM Cortex (insulin pump microcontroller)
- **Memory:** <10 MB for model + buffers
- **Inference latency:** <100 ms (5-min decision interval)

### Model Compression (for embedded systems)

```python
# Quantization to 8-bit integers
import torch.quantization as quant
quantized_model = quant.quantize_dynamic(agent.actor, ...)
```

### Safety Requirements

1. **Watchdog:** If network inference fails → PID takeover
2. **Bounds checking:** Double-check insulin before delivery
3. **Logging:** Store all decisions for post-hoc analysis
4. **Override:** Patient can emergency pause pump

### Regulatory Path (FDA)

- Classify as Software as a Medical Device (SaMD)
- Pre-clinical validation: 100+ virtual patients
- Clinical trial: 30-50 real patients
- Continuous post-market surveillance

---

## References

1. **Wang, S., & Gu, W.** (2024). "An Improved Strategy for Blood Glucose Control Using Multi-Step Deep Reinforcement Learning." *arXiv:2403.07566*
   
2. **Emerson, H., et al.** (2023). "Offline Reinforcement Learning for Safer Blood Glucose Control." *arXiv:2204.03376*

3. **Patent WO 2018/011766 A1** - Estimation of Insulin Based on RL (Universität Bern)

4. **Patent US 10,646,650 B2** - Multivariable Artificial Pancreas (Illinois Institute of Technology)

5. **Patent US 2022/0088304 A1** - Intraperitoneal Closed-Loop Control (Harvard College)

---

## Troubleshooting

### Q: Agent not learning (reward stays constant)
**A:** Check reward function is working. Print `rewards['total']` each step.

### Q: Frequent hypoglycemia episodes
**A:** Increase `w_hypo` weight in RewardShaper. Add more training episodes.

### Q: Safety layer blocking too many actions
**A:** RL agent is proposing unsafe doses. Reduce learning rate, add safety penalty to reward.

### Q: GPU out of memory
**A:** Reduce `BATCH_SIZE` from 32 → 16. Or use CPU (`device='cpu'`).

---

## Contributing

To extend this project:
1. Implement routine learner (pattern detection)
2. Add SimGlucose integration (replace mock environment)
3. Implement offline RL variant (TD3-BC)
4. Add multi-patient training (transfer learning)
5. Implement model distillation for embedded devices

---

## License

This is an educational implementation for the course ME 696 (Biomedical Devices & Systems).

For clinical use, consult regulatory bodies (FDA, EMA, etc.) and follow established protocols for medical device approval.

---

**Last Updated:** February 2026

**Authors:** Ayush Kumar, Ayush Savar, Arush Jain, Tarun Raj 