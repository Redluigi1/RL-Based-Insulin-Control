import numpy as np

from state_encoder import StateEncoder
from safety_layer import SafetyLayer
from reward_shaper import RewardShaper
from actor_critic_agent import ActorCriticAgent
from trash.simglucose_env import SimGlucoseEnvWrapper


def compute_metrics(glucose):

    glucose = np.array(glucose)

    tir = np.mean((glucose >= 70) & (glucose <= 180)) * 100
    tbr = np.mean(glucose < 70) * 100
    tar = np.mean(glucose > 180) * 100

    return tir, tbr, tar


def run_episode(agent, env, encoder, safety):

    glucose = env.reset()

    encoder.reset()
    safety.reset()

    history = []
    prev_action = 0

    for step in range(env.max_steps):

        time_hours = (step * 5) / 60

        state = encoder.encode(
            glucose=glucose,
            time_hours=time_hours,
            active_insulin=env.active_insulin,
            activity_level=0,
            routine_context={"meal_likely":0},
        )

        state_vec = encoder.flatten_state(state)

        action = agent.select_action(state_vec, deterministic=True)

        safe_action, _, _ = safety.evaluate_action(
            rl_proposed_insulin=action,
            glucose=glucose,
            active_insulin=env.active_insulin,
            previous_action=prev_action,
        )

        glucose, done = env.step(safe_action)

        history.append(glucose)

        prev_action = safe_action

        if done:
            break

    return history


def main():

    encoder = StateEncoder(glucose_history_size=11)
    state_dim = encoder.get_state_dim()

    agent = ActorCriticAgent(state_dim)

    agent.load("logs/agent_episode_50.pt")

    safety = SafetyLayer()
    env = SimGlucoseEnvWrapper()

    glucose_history = run_episode(agent, env, encoder, safety)

    tir, tbr, tar = compute_metrics(glucose_history)

    print("Clinical metrics")
    print("TIR:", tir)
    print("TBR:", tbr)
    print("TAR:", tar)


if __name__ == "__main__":
    main()