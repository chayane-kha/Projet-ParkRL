"""
config.py — Paramètres globaux du projet ParkRL.

Ce module centralise toutes les constantes physiques (en mètres, secondes et radians),
la géométrie du parking, les bornes de commande du véhicule ainsi que les paramètres
de la fonction de récompense (calibrables à partir des données réelles).
"""

import math
from pathlib import Path

# Répertoire racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "ppo_parking_default.zip"

# ---------------------------------------------------------------------------
# 1. Dimensions du parking (en mètres)
# ---------------------------------------------------------------------------
WIDTH = 16.0
HEIGHT = 16.0
WALL_THICKNESS = 0.25

# Rangée de places de parking en bataille en haut du parking
PARKING_ROW_Y = 13.0
SLOT_WIDTH = 2.7
SLOT_LENGTH = 4.8

# Position et orientation de la place cible libre (au centre de la rangée)
TARGET_X = 8.0
TARGET_Y = PARKING_ROW_Y
TARGET_THETA = math.pi / 2.0  # Nez orienté vers le haut (+90°)

# Centres X des véhicules déjà garés entourant la place cible
PARKED_CARS_X = [2.6, 5.3, 10.7, 13.4]

# Zone d'apparition aléatoire du véhicule dans l'allée de circulation
SPAWN_X_RANGE = (4.2, 11.8)
SPAWN_Y_RANGE = (4.8, 7.2)
SPAWN_THETA_RANGE = (-math.pi / 4.0, 3.0 * math.pi / 4.0)

# ---------------------------------------------------------------------------
# 2. Dimensions et cinématique du véhicule (Modèle Bicyclette)
# ---------------------------------------------------------------------------
CAR_LENGTH = 4.2        # Longueur totale du véhicule (m)
CAR_WIDTH = 1.9         # Largeur totale du véhicule (m)
WHEELBASE = 2.6         # Empattement L entre essieux avant et arrière (m)

DT = 0.2                # Pas de simulation (s)
MAX_STEPS = 250         # Durée maximale d'un épisode (250 * 0.2s = 50s)

MIN_SPEED = -2.5        # Vitesse maximale en marche arrière (m/s)
MAX_SPEED = 3.5         # Vitesse maximale en marche avant (m/s)
MAX_STEER_DEG = 35.0    # Angle de braquage maximal du volant (degrés)
MAX_STEER_RAD = math.radians(MAX_STEER_DEG)

# ---------------------------------------------------------------------------
# 3. Capteurs laser (LiDAR simplifié)
# ---------------------------------------------------------------------------
NUM_RAYS = 16           # Nombre de rayons laser répartis sur 360° (précision accrue)
RAY_MAX_DIST = 6.0      # Portée maximale des capteurs laser (m)

# ---------------------------------------------------------------------------
# 4. Critères de réussite du stationnement & anti-immobilisme
# ---------------------------------------------------------------------------
POS_TOLERANCE = 0.80                  # Distance max au centre de la place (m)
ANGLE_TOLERANCE_DEG = 10.0            # Écart angulaire max (degrés)
ANGLE_TOLERANCE_RAD = math.radians(ANGLE_TOLERANCE_DEG)
SPEED_TOLERANCE = 0.55                # Vitesse quasi nulle requise (m/s)
MAX_IDLE_STEPS = 20                   # Pas consécutifs à l'arrêt avant arrêt pour immobilisme (4s)

# ---------------------------------------------------------------------------
# 5. Pénalités et récompenses (calibrables par l'analyse de données)
# ---------------------------------------------------------------------------
PENALTY_STEP = -0.05            # Coût temporel par pas de simulation (dt = 0.2s)
PENALTY_REVERSE = -0.25         # Pénalité accrue en marche arrière (risque d'accident)
PENALTY_STEER_CHANGE = -0.10    # Pénalité pour coup de volant brusque (fluidité)
REWARD_DISTANCE_FACTOR = 1.5    # Récompense dense proportionnelle au rapprochement
PENALTY_COLLISION = -100.0      # Pénalité en cas de collision géométrique
PENALTY_IDLE = -40.0            # Pénalité si le véhicule reste immobilisé hors de la place
REWARD_PARKED = +100.0          # Récompense terminale pour stationnement réussi
