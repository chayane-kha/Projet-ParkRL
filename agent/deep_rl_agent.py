"""
agent/deep_rl_agent.py — Agent de Deep Reinforcement Learning (PPO via Stable-Baselines3).

Ce module encapsule le modèle PPO("MlpPolicy", env, ...) pour :
- Entraîner l'agent tout en collectant l'historique des récompenses par épisode.
- Prédire l'action déterministe à partir d'une observation à `NUM_RAYS + 4` dimensions.
- Extraire la carte de valeur critique (Value Function Heatmap) sur une grille 2D.
- Sauvegarder et charger des modèles entraînés au format `.zip`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

import config
from env.parking_env import ContinuousParkingEnv


class _TrainingProgressCallback(BaseCallback):
    """Callback Stable-Baselines3 pour suivre la progression, le curriculum et les récompenses."""

    def __init__(
        self,
        total_timesteps: int,
        raw_env: Optional[ContinuousParkingEnv] = None,
        progress_callback: Optional[Callable[[int, int, List[float]], None]] = None,
        update_freq: int = 1000,
    ) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.raw_env = raw_env
        self.progress_callback = progress_callback
        self.update_freq = update_freq
        self.episode_rewards: List[float] = []
        self.episode_successes: List[bool] = []
        self._last_reported = 0

    def _on_step(self) -> bool:
        # Progression fluide du curriculum sur les premiers 50 % de l'entraînement
        if self.raw_env is not None:
            curr_prog = min(1.0, self.num_timesteps / max(1.0, 0.50 * self.total_timesteps))
            self.raw_env.set_curriculum_progress(curr_prog)

        infos = self.locals.get("infos", [])
        for info in infos:
            # `Monitor` injecte "episode": {"r": ..., "l": ..., "t": ...} à la fin de chaque épisode
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_successes.append(bool(info.get("is_success", False)))

        if (
            self.progress_callback is not None
            and (self.num_timesteps - self._last_reported >= self.update_freq
                 or self.num_timesteps >= self.total_timesteps)
        ):
            self._last_reported = self.num_timesteps
            self.progress_callback(
                min(self.num_timesteps, self.total_timesteps),
                self.total_timesteps,
                self.episode_rewards,
            )
        return True


class DeepParkingAgent:
    """
    Gestionnaire complet de l'agent PPO pour le stationnement automatique.
    """

    def __init__(
        self,
        env: ContinuousParkingEnv,
        learning_rate: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.raw_env = env
        self.monitored_env = Monitor(env, info_keywords=("is_success",))
        self.vec_env = VecNormalize(
            DummyVecEnv([lambda: self.monitored_env]),
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
        )
        self.learning_rate = learning_rate
        self.device = device
        self.reward_history: List[float] = []
        self.success_history: List[bool] = []

        # Architecture Actor-Critic à 2 couches cachées de 128 neurones
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            log_std_init=-0.4,
        )

        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            learning_rate=self.learning_rate,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            policy_kwargs=policy_kwargs,
            device=self.device,
        )

    def train(
        self,
        total_timesteps: int = 100_000,
        progress_callback: Optional[Callable[[int, int, List[float]], None]] = None,
    ) -> List[float]:
        """
        Lance l'entraînement PPO et enregistre l'historique des récompenses d'épisodes.

        Paramètres
        ----------
        total_timesteps : nombre total d'interactions avec l'environnement.
        progress_callback : fonction optionnelle `(step, total, reward_history)` appelée
            périodiquement pour mettre à jour une barre de progression (ex. Streamlit).
        """
        update_freq = max(500, total_timesteps // 40)
        self.raw_env.set_curriculum_progress(0.0)
        cb = _TrainingProgressCallback(
            total_timesteps=total_timesteps,
            raw_env=self.raw_env,
            progress_callback=progress_callback,
            update_freq=update_freq,
        )
        try:
            self.model.learn(total_timesteps=total_timesteps, callback=cb, reset_num_timesteps=False)
        finally:
            # Toujours remettre l'environnement en difficulté complète après l'entraînement
            self.raw_env.set_curriculum_progress(1.0)
        self.reward_history.extend(cb.episode_rewards)
        self.success_history.extend(cb.episode_successes)
        return self.reward_history

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """
        Retourne l'action déterministe normalisée `[accel, braquage]` dans `[-1, 1]^2`.

        Pour obtenir les grandeurs physiques `(vitesse_m_s, braquage_rad)`, utiliser
        `ContinuousParkingEnv.decode_action(action)` ou `predict_controls(observation)`.
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def predict_controls(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, float, float]:
        """Retourne `(action_norm, vitesse_m_s, braquage_rad)`."""
        action = self.predict(observation, deterministic=deterministic)
        v, steer_rad = self.raw_env.decode_action(action)
        return action, v, steer_rad

    def evaluate_grid_values(
        self,
        env: Optional[ContinuousParkingEnv] = None,
        grid_res: int = 15,
        theta: float = config.TARGET_THETA,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interroge le réseau Critique (`model.policy.predict_values`) sur une grille 2D
        de positions `(grid_res x grid_res)` dans le parking.

        Retourne
        --------
        xs : vecteur 1D des coordonnées X (taille `grid_res`)
        ys : vecteur 1D des coordonnées Y (taille `grid_res`)
        value_grid : matrice 2D de forme `(grid_res, grid_res)` indexée `[iy, ix]`
                     contenant la valeur V(s) estimée par le Critique PPO.
        """
        target_env = env if env is not None else self.raw_env
        xs = np.linspace(0.6, target_env.width - 0.6, grid_res)
        ys = np.linspace(0.6, target_env.height - 0.6, grid_res)

        obs_batch = []
        for y in ys:
            for x in xs:
                obs = target_env.compute_observation_at(
                    float(x), float(y), float(theta), v=0.0, cache_rays=False
                )
                obs_batch.append(obs)

        obs_tensor = torch.as_tensor(
            np.asarray(obs_batch, dtype=np.float32), device=self.model.device
        )
        with torch.no_grad():
            values = self.model.policy.predict_values(obs_tensor)

        value_grid = values.squeeze(-1).cpu().numpy().reshape(grid_res, grid_res)
        return xs, ys, value_grid

    def save(self, filepath: Union[str, Path] = config.DEFAULT_MODEL_PATH) -> Path:
        """Sauvegarde le modèle PPO au format `.zip`."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        return path

    def load(self, filepath: Union[str, Path] = config.DEFAULT_MODEL_PATH) -> None:
        """Charge un modèle PPO pré-entraîné depuis un fichier `.zip`."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        self.model = PPO.load(str(path), env=self.vec_env, device=self.device)
