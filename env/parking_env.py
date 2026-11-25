"""
env/parking_env.py — Environnement continu Gymnasium pour le stationnement automatique.

Cet environnement implémente :
- Un parking 2D continu (16m x 16m) délimité par des murs et contenant des voitures garées (Shapely).
- Un véhicule régi par le modèle cinématique bicyclette.
- `NUM_RAYS` (16) capteurs laser (LiDAR 360°) calculés par intersection géométrique avec Shapely.
- Un espace d'actions continu (vitesse cible et angle de braquage) et `NUM_RAYS + 4` (20) observations normalisées.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

import config


def wrap_angle(angle: float) -> float:
    """Normalise un angle en radians dans l'intervalle [-pi, pi]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def create_oriented_car_polygon(
    x: float,
    y: float,
    theta: float,
    length: float = config.CAR_LENGTH,
    width: float = config.CAR_WIDTH,
) -> Polygon:
    """
    Construit le polygone Shapely orienté représentant l'emprise du véhicule.

    Le calcul direct des 4 sommets par rotation matricielle 2D évite les appels
    coûteux à `shapely.affinity` lors des boucles d'entraînement RL.
    """
    half_l = length * 0.5
    half_w = width * 0.5
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Coins locaux : avant-droit, avant-gauche, arrière-gauche, arrière-droit
    local_corners = (
        (half_l, -half_w),
        (half_l, half_w),
        (-half_l, half_w),
        (-half_l, -half_w),
    )
    world_corners = [
        (x + lx * cos_t - ly * sin_t, y + lx * sin_t + ly * cos_t)
        for lx, ly in local_corners
    ]
    return Polygon(world_corners)


class ContinuousParkingEnv(gym.Env):
    """
    Environnement Gymnasium continu simulant le stationnement en bataille d'un véhicule.

    Observations (NUM_RAYS + 4 = 20 valeurs normalisées dans [-1, 1]) :
        - [0..15] : 16 distances LiDAR normalisées dans [0, 1] (divisées par RAY_MAX_DIST)
        - [16..17] : (dx_rel, dy_rel) vecteur vers la place cible dans le repère véhicule
        - [18]     : différence d'orientation normalisée dans [-1, 1]
        - [19]     : vitesse actuelle normalisée dans [-1, 1]

    Actions (2 valeurs continues dans [-1, 1]) :
        - action[0] : consigne de vitesse mappée sur [MIN_SPEED, MAX_SPEED] = [-2.5, +3.5] m/s
        - action[1] : angle de braquage mappé sur [-35°, +35°]
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        penalty_step: float = config.PENALTY_STEP,
        penalty_reverse: float = config.PENALTY_REVERSE,
        penalty_steer_change: float = config.PENALTY_STEER_CHANGE,
        reward_distance_factor: float = config.REWARD_DISTANCE_FACTOR,
        penalty_collision: float = config.PENALTY_COLLISION,
        reward_parked: float = config.REWARD_PARKED,
        max_steps: int = config.MAX_STEPS,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.dt = config.DT
        self.wheelbase = config.WHEELBASE
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Paramètres de récompense (modifiables dynamiquement via l'UI)
        self.penalty_step = float(penalty_step)
        self.penalty_reverse = float(penalty_reverse)
        self.penalty_steer_change = float(penalty_steer_change)
        self.reward_distance_factor = float(reward_distance_factor)
        self.penalty_collision = float(penalty_collision)
        self.reward_parked = float(reward_parked)

        # Espaces Gymnasium
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.NUM_RAYS + 4,), dtype=np.float32
        )

        # Géométrie statique du parking (murs + voitures garées + place cible)
        self._build_parking_geometry()

        # Angles relatifs des 8 capteurs laser répartis autour du véhicule
        self.ray_rel_angles = np.linspace(
            0.0, 2.0 * math.pi, config.NUM_RAYS, endpoint=False
        )

        # État interne du véhicule : (x, y, theta, v, steer_angle)
        self.x = 8.0
        self.y = 6.0
        self.theta = 0.0
        self.v = 0.0
        self.steer_angle = 0.0

        self.steps = 0
        self.idle_steps = 0
        self.total_reward = 0.0
        self.reverse_time = 0.0
        self.min_obstacle_dist_m = 0.0
        self.curriculum_progress = 1.0  # 1.0 = difficulté complète (allée entière)
        self.last_rays: List[Tuple[float, Tuple[float, float]]] = []
        self.trajectory: List[Dict[str, float]] = []
        self.status = "En cours"

        # Cache pour le rendu Matplotlib
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    def _build_parking_geometry(self) -> None:
        """Construit les polygones Shapely des murs, obstacles et place cible."""
        t = config.WALL_THICKNESS
        w, h = self.width, self.height

        # 4 murs extérieurs délimitant l'aire [0, WIDTH] x [0, HEIGHT]
        self.walls: List[Polygon] = [
            box(0.0, 0.0, w, t),           # Mur bas
            box(0.0, h - t, w, h),         # Mur haut
            box(0.0, 0.0, t, h),           # Mur gauche
            box(w - t, 0.0, w, h),         # Mur droit
        ]

        # Voitures déjà garées en bataille sur la rangée du haut (orientées verticalement)
        self.parked_cars: List[Polygon] = [
            create_oriented_car_polygon(
                cx,
                config.PARKING_ROW_Y,
                config.TARGET_THETA,
                length=config.CAR_LENGTH,
                width=config.CAR_WIDTH,
            )
            for cx in config.PARKED_CARS_X
        ]

        # Union préparée pour accélérer les tests d'intersection et le raycasting
        self.obstacles_list: List[Polygon] = self.walls + self.parked_cars
        self.obstacles_union = unary_union(self.obstacles_list)
        self.prep_obstacles = prep(self.obstacles_union)

        # Boîte représentant la place cible libre
        self.target_x = config.TARGET_X
        self.target_y = config.TARGET_Y
        self.target_theta = config.TARGET_THETA
        self.target_box: Polygon = box(
            self.target_x - config.SLOT_WIDTH * 0.5,
            self.target_y - config.SLOT_LENGTH * 0.5,
            self.target_x + config.SLOT_WIDTH * 0.5,
            self.target_y + config.SLOT_LENGTH * 0.5,
        )

    def update_penalties(
        self,
        penalty_reverse: Optional[float] = None,
        penalty_steer_change: Optional[float] = None,
        reward_distance_factor: Optional[float] = None,
    ) -> None:
        """Met à jour dynamiquement les coefficients de pénalité de l'environnement."""
        if penalty_reverse is not None:
            self.penalty_reverse = float(penalty_reverse)
        if penalty_steer_change is not None:
            self.penalty_steer_change = float(penalty_steer_change)
        if reward_distance_factor is not None:
            self.reward_distance_factor = float(reward_distance_factor)

    @staticmethod
    def decode_action(action: np.ndarray) -> Tuple[float, float]:
        """
        Convertit une action normalisée [-1, 1]^2 en commandes physiques (v, steer_rad).

        - action[0] >= 0 -> [0, MAX_SPEED], action[0] < 0 -> [MIN_SPEED, 0].
        - action[1] -> [-MAX_STEER_RAD, +MAX_STEER_RAD].
        """
        a_vel = float(np.clip(action[0], -1.0, 1.0))
        a_steer = float(np.clip(action[1], -1.0, 1.0))

        if a_vel >= 0.0:
            target_v = a_vel * config.MAX_SPEED
        else:
            target_v = a_vel * abs(config.MIN_SPEED)

        target_steer = a_steer * config.MAX_STEER_RAD
        return target_v, target_steer

    def _cast_rays(
        self, x: float, y: float, theta: float
    ) -> List[Tuple[float, Tuple[float, float]]]:
        """
        Lance les 8 rayons laser depuis (x, y) via l'API vectorisée de Shapely 2.x.

        Retourne une liste de tuples `(distance_metres, (end_x, end_y))`.
        """
        angles = theta + self.ray_rel_angles
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        end_xs = x + config.RAY_MAX_DIST * cos_a
        end_ys = y + config.RAY_MAX_DIST * sin_a

        # Construction vectorisée des 8 segments (N, 2, 2) en un seul appel Shapely C
        coords = np.empty((config.NUM_RAYS, 2, 2), dtype=np.float64)
        coords[:, 0, 0] = x
        coords[:, 0, 1] = y
        coords[:, 1, 0] = end_xs
        coords[:, 1, 1] = end_ys

        rays = shapely.linestrings(coords)
        inters = shapely.intersection(self.obstacles_union, rays)
        empty_mask = shapely.is_empty(inters)

        origin = Point(x, y)
        dists = np.full(config.NUM_RAYS, config.RAY_MAX_DIST, dtype=np.float64)
        if not np.all(empty_mask):
            hit_idx = np.where(~empty_mask)[0]
            hit_dists = shapely.distance(origin, inters[hit_idx])
            dists[hit_idx] = np.minimum(config.RAY_MAX_DIST, hit_dists)

        hit_xs = x + dists * cos_a
        hit_ys = y + dists * sin_a
        return [
            (float(dists[i]), (float(hit_xs[i]), float(hit_ys[i])))
            for i in range(config.NUM_RAYS)
        ]

    def compute_observation_at(
        self,
        x: float,
        y: float,
        theta: float,
        v: float = 0.0,
        cache_rays: bool = False,
    ) -> np.ndarray:
        """
        Calcule le vecteur d'observation normalisé (12 dimensions) pour un état donné.

        Utilisable à la fois dans `step()`/`reset()` et pour évaluer la fonction
        de valeur Critique sur une grille 2D sans altérer l'état de la simulation.
        """
        rays = self._cast_rays(x, y, theta)
        if cache_rays:
            self.last_rays = rays

        ray_dists_norm = np.array(
            [dist / config.RAY_MAX_DIST for dist, _ in rays], dtype=np.float32
        )

        # Vecteur distance vers la place cible dans le repère local de la voiture
        dx_world = self.target_x - x
        dy_world = self.target_y - y
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        scale = 8.0  # Résolution accrue près de la place cible (normalisé dans [-1, 1])

        dx_rel = np.clip((cos_t * dx_world + sin_t * dy_world) / scale, -1.0, 1.0)
        dy_rel = np.clip((-sin_t * dx_world + cos_t * dy_world) / scale, -1.0, 1.0)

        # Différence angulaire normalisée dans [-1, 1]
        angle_diff = wrap_angle(self.target_theta - theta) / math.pi

        # Vitesse normalisée dans [-1, 1]
        if v >= 0.0:
            v_norm = np.clip(v / config.MAX_SPEED, -1.0, 1.0)
        else:
            v_norm = np.clip(v / abs(config.MIN_SPEED), -1.0, 1.0)

        obs = np.concatenate(
            [
                ray_dists_norm,
                np.array([dx_rel, dy_rel, angle_diff, v_norm], dtype=np.float32),
            ]
        ).astype(np.float32)
        return obs

    def _distance_to_target(self, x: float, y: float) -> float:
        """Distance euclidienne au centre de la place cible (m)."""
        return math.hypot(self.target_x - x, self.target_y - y)

    def _is_parked(self, x: float, y: float, theta: float, v: float) -> bool:
        """Vérifie si le véhicule est centré, aligné (< 10°) et quasi à l'arrêt."""
        dist = self._distance_to_target(x, y)
        angle_err = abs(wrap_angle(self.target_theta - theta))
        return (
            dist <= config.POS_TOLERANCE
            and angle_err <= config.ANGLE_TOLERANCE_RAD
            and abs(v) <= config.SPEED_TOLERANCE
        )

    def _check_collision(self, car_poly: Polygon) -> bool:
        """Détecte toute collision géométrique avec les murs ou voitures garées."""
        return bool(self.prep_obstacles.intersects(car_poly))

    def set_curriculum_progress(self, progress: float) -> None:
        """Définit le niveau de difficulté du curriculum [0.0 (proche) -> 1.0 (allée complète)]."""
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def _sample_initial_pose(self) -> Tuple[float, float, float]:
        """
        Tire une position initiale valide sans collision selon `curriculum_progress`.

        - progress = 0.0 : entrée proche de la place (apprentissage du freinage et de l'alignement).
        - progress = 1.0 : allée complète (gauche, centre, droite) avec orientations réalistes.
        """
        p = self.curriculum_progress
        # Conserve 15 % de départs proches en fin de curriculum pour entretenir la précision d'arrêt
        use_near = bool(p < 0.15 or self.np_random.random() < (0.15 + 0.85 * (1.0 - p)))

        for _ in range(50):
            if use_near:
                y = float(self.np_random.uniform(9.8, 12.2))
                x = float(self.np_random.uniform(7.4, 8.6))
                theta = wrap_angle(
                    float(
                        self.np_random.uniform(
                            config.TARGET_THETA - 0.25 - 0.20 * p,
                            config.TARGET_THETA + 0.25 + 0.20 * p,
                        )
                    )
                )
            else:
                y_min = 9.5 - (9.5 - config.SPAWN_Y_RANGE[0]) * p
                y = float(self.np_random.uniform(y_min, 10.2))
                x_span = 0.8 + 3.2 * p
                x = float(
                    self.np_random.uniform(
                        max(config.SPAWN_X_RANGE[0], config.TARGET_X - x_span),
                        min(config.SPAWN_X_RANGE[1], config.TARGET_X + x_span),
                    )
                )
                # Orientation réaliste dans l'allée, avec 15 % de départs désaxés pour
                # entraîner le véhicule à reculer et se replacer
                if p > 0.3 and self.np_random.random() < 0.15:
                    theta = wrap_angle(float(self.np_random.uniform(-math.pi, math.pi)))
                else:
                    if x < config.TARGET_X - 0.5:
                        # Vient de la gauche : entre horizontal (0) et face à la rangée (pi/2)
                        base_theta = 0.25 * math.pi
                    elif x > config.TARGET_X + 0.5:
                        # Vient de la droite : entre face à la rangée (pi/2) et horizontal gauche (pi)
                        base_theta = 0.75 * math.pi
                    else:
                        base_theta = config.TARGET_THETA
                    theta_spread = 0.25 + 0.65 * p
                    theta = wrap_angle(
                        float(self.np_random.uniform(base_theta - theta_spread, base_theta + theta_spread))
                    )

            car_poly = create_oriented_car_polygon(x, y, theta)
            if not self._check_collision(car_poly):
                return x, y, theta
        return 8.0, 5.5, math.pi / 2.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Réinitialise l'environnement pour un nouvel essai de stationnement."""
        super().reset(seed=seed)

        if options and all(k in options for k in ("x", "y", "theta")):
            self.x = float(options["x"])
            self.y = float(options["y"])
            self.theta = wrap_angle(float(options["theta"]))
        else:
            self.x, self.y, self.theta = self._sample_initial_pose()

        self.v = 0.0
        self.steer_angle = 0.0
        self.steps = 0
        self.idle_steps = 0
        self.total_reward = 0.0
        self.reverse_time = 0.0
        self.status = "En cours"

        car_poly = create_oriented_car_polygon(self.x, self.y, self.theta)
        self.min_obstacle_dist_m = float(car_poly.distance(self.obstacles_union))

        self.trajectory = [
            {
                "step": 0,
                "t": 0.0,
                "x": self.x,
                "y": self.y,
                "theta": self.theta,
                "v": self.v,
                "steer": self.steer_angle,
                "min_dist_cm": self.min_obstacle_dist_m * 100.0,
            }
        ]

        obs = self.compute_observation_at(
            self.x, self.y, self.theta, self.v, cache_rays=True
        )
        info = self._build_info(collision=False, parked=False)
        return obs, info

    def _compute_potential(self, x: float, y: float, theta: float, v: float) -> float:
        """
        Fonction de potentiel Φ(s) guidant le véhicule sans biais de stationnement :
        - Utilise une distance guidée contournant les voitures voisines (via le point
          d'alignement devant la place libre à y = 9.2 m) lorsque le véhicule est décalé
          latéralement, puis bascule sur la distance directe dans l'axe de la place.
        - Encourage l'alignement angulaire progressif et la décélération à l'arrivée.
        """
        direct_dist = self._distance_to_target(x, y)
        staging_y = 9.2
        via_staging_dist = (
            math.hypot(self.target_x - x, staging_y - y)
            + abs(self.target_y - staging_y)
        )
        # Poids de transition lisse : 0 dans l'axe de la place (|x - 8| < 0.8), 1 en allée latérale
        lateral_offset = abs(x - self.target_x)
        if y >= 10.4 and lateral_offset < 1.3:
            blend = 0.0
        else:
            blend = float(np.clip((lateral_offset - 0.6) / 1.2, 0.0, 1.0))

        effective_dist = (1.0 - blend) * direct_dist + blend * via_staging_dist
        angle_err = abs(wrap_angle(self.target_theta - theta))
        align_weight = 2.0 * math.exp(-direct_dist / 5.0)
        stop_weight = 1.5 * math.exp(-direct_dist / 1.0)
        return (
            -self.reward_distance_factor * effective_dist
            - align_weight * angle_err
            - stop_weight * abs(v)
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Exécute un pas de simulation (dt = 0.2s) selon le modèle bicyclette."""
        prev_potential = self._compute_potential(self.x, self.y, self.theta, self.v)
        prev_steer = self.steer_angle

        # Décodage des commandes continues avec inertie réaliste de vitesse
        target_v, target_steer = self.decode_action(action)
        self.v = 0.4 * self.v + 0.6 * target_v
        self.steer_angle = target_steer

        # Cinématique du modèle bicyclette
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta = wrap_angle(
            self.theta + (self.v / self.wheelbase) * math.tan(self.steer_angle) * self.dt
        )
        self.steps += 1

        # Vérifications géométriques via Shapely
        car_poly = create_oriented_car_polygon(self.x, self.y, self.theta)
        collision = self._check_collision(car_poly)
        parked = (not collision) and self._is_parked(
            self.x, self.y, self.theta, self.v
        )

        # Mise à jour de la marge de sécurité minimale avec les obstacles
        current_obs_dist = 0.0 if collision else float(car_poly.distance(self.obstacles_union))
        self.min_obstacle_dist_m = min(self.min_obstacle_dist_m, current_obs_dist)

        # Calcul de la fonction de récompense basée sur le potentiel et les coûts métier
        curr_potential = self._compute_potential(self.x, self.y, self.theta, self.v)
        curr_dist = self._distance_to_target(self.x, self.y)

        reward = self.penalty_step
        reward += (curr_potential - prev_potential)

        # Pénalité lisse d'immobilisme loin de la place + détection de blocage prolongé
        if curr_dist > config.POS_TOLERANCE and abs(self.v) < 0.25:
            self.idle_steps += 1
        else:
            self.idle_steps = 0

        if curr_dist > 2.0:
            reward -= 0.40 * max(0.0, 1.0 - abs(self.v))

        # Pénalité accrue en marche arrière
        if self.v < -0.05:
            reward += self.penalty_reverse
            self.reverse_time += self.dt

        # Pénalité pour variation brusque de braquage (fluidité de conduite)
        steer_change_norm = abs(self.steer_angle - prev_steer) / config.MAX_STEER_RAD
        reward += self.penalty_steer_change * steer_change_norm

        terminated = False
        if collision:
            reward += self.penalty_collision
            terminated = True
            self.status = "Collision"
        elif parked:
            reward += self.reward_parked
            terminated = True
            self.status = "Garé avec succès"
        elif self.idle_steps >= config.MAX_IDLE_STEPS:
            reward += config.PENALTY_IDLE
            terminated = True
            self.status = "Immobilisé"

        truncated = (self.steps >= self.max_steps) and not terminated
        if truncated:
            self.status = "Temps dépassé"

        self.total_reward += reward

        # Enregistrement dans l'historique de trajectoire
        self.trajectory.append(
            {
                "step": self.steps,
                "t": self.steps * self.dt,
                "x": self.x,
                "y": self.y,
                "theta": self.theta,
                "v": self.v,
                "steer": self.steer_angle,
                "min_dist_cm": current_obs_dist * 100.0,
            }
        )

        obs = self.compute_observation_at(
            self.x, self.y, self.theta, self.v, cache_rays=True
        )
        info = self._build_info(collision=collision, parked=parked)
        return obs, float(reward), terminated, truncated, info

    def _build_info(self, collision: bool, parked: bool) -> Dict[str, Any]:
        """Construit le dictionnaire d'informations télémétriques de l'étape."""
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "v": self.v,
            "steer_deg": math.degrees(self.steer_angle),
            "steps": self.steps,
            "elapsed_time_s": self.steps * self.dt,
            "reverse_time_s": self.reverse_time,
            "min_obstacle_dist_cm": self.min_obstacle_dist_m * 100.0,
            "total_reward": self.total_reward,
            "collision": collision,
            "is_success": parked,
            "status": self.status,
        }

    def render(self) -> Optional[np.ndarray]:
        """Génère une vue 2D Matplotlib du parking, du véhicule et des rayons LiDAR."""
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6), dpi=110)

        ax = self._ax
        ax.clear()
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(0.0, self.height)
        ax.set_aspect("equal")
        ax.set_facecolor("#1e242b")

        # Murs extérieurs
        for wall in self.walls:
            wx, wy = wall.exterior.xy
            ax.fill(wx, wy, color="#495057", zorder=2)

        # Place cible en vert
        tx, ty = self.target_box.exterior.xy
        ax.fill(tx, ty, color="#2ecc71", alpha=0.28, edgecolor="#27ae60", linewidth=2, zorder=1)

        # Voitures déjà garées
        for pcar in self.parked_cars:
            px, py = pcar.exterior.xy
            ax.fill(px, py, color="#5c6bc0", edgecolor="#c5cae9", linewidth=1.2, zorder=3)

        # Trajectoire parcourue
        if len(self.trajectory) > 1:
            xs = [pt["x"] for pt in self.trajectory]
            ys = [pt["y"] for pt in self.trajectory]
            ax.plot(xs, ys, "--", color="#f1c40f", linewidth=1.8, zorder=4)

        # Rayons LiDAR (vert -> rouge selon la proximité)
        for dist, (hx, hy) in self.last_rays:
            ratio = dist / config.RAY_MAX_DIST
            ray_color = plt.cm.RdYlGn(ratio)
            ax.plot([self.x, hx], [self.y, hy], color=ray_color, linewidth=1.3, alpha=0.85, zorder=5)

        # Véhicule contrôlé
        car_poly = create_oriented_car_polygon(self.x, self.y, self.theta)
        cx, cy = car_poly.exterior.xy
        car_patch = MplPolygon(
            list(zip(cx, cy)),
            closed=True,
            facecolor="#e74c3c" if self.status == "Collision" else "#3498db",
            edgecolor="#ffffff",
            linewidth=1.6,
            zorder=6,
        )
        ax.add_patch(car_patch)

        # Flèche de direction du véhicule
        ax.arrow(
            self.x,
            self.y,
            1.4 * math.cos(self.theta),
            1.4 * math.sin(self.theta),
            head_width=0.35,
            head_length=0.35,
            fc="#ffffff",
            ec="#ffffff",
            zorder=7,
        )

        ax.set_title(f"ParkRL — {self.status} (t = {self.steps * self.dt:.1f}s)", color="#111111")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        if self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.asarray(self._fig.canvas.buffer_rgba())
            return buf[:, :, :3].copy()
        if self.render_mode == "human":
            plt.pause(0.01)
        return None

    def close(self) -> None:
        """Ferme les ressources graphiques Matplotlib."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
