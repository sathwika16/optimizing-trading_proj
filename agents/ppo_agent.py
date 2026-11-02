"""Stable-Baselines3 PPO integration."""

from __future__ import annotations

from pathlib import Path


def train_ppo(
    env,
    timesteps: int,
    seed: int,
    model_path: str | Path,
    learning_rate: float = 3e-4,
):
    """Train and persist PPO; the dependency error is intentionally explicit."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "Install stable-baselines3 and gymnasium to train PPO."
        ) from exc
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, learning_rate=learning_rate)
    model.learn(total_timesteps=timesteps)
    model.save(str(model_path))
    return model


def predict_actions(model, env) -> list[int]:
    """Run a deterministic policy over an environment."""
    observation, _ = env.reset()
    actions: list[int] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        actions.append(int(action))
        observation, _, terminated, truncated, _ = env.step(int(action))
    return actions
