# ParkRL — Stationnement Automatique par Deep Reinforcement Learning

**ParkRL** est une application Python complète permettant d'entraîner, d'évaluer et de visualiser un agent de **Deep Reinforcement Learning (PPO via Stable-Baselines3)** capable de garer un véhicule de manière fluide dans un parking simulé en espace continu (**Gymnasium + Shapely**).

---

## Architecture du projet

```text
parkrl/
├── requirements.txt          # Dépendances du projet
├── config.py                 # Paramètres globaux (dimensions, cinématique, pénalités)
├── env/
│   ├── __init__.py
│   └── parking_env.py        # Environnement continu Gymnasium (modèle bicyclette + LiDAR Shapely)
├── agent/
│   ├── __init__.py
│   └── deep_rl_agent.py      # Entraînement, évaluation Critique et sauvegarde/chargement PPO
├── data_analysis/
│   ├── __init__.py
│   └── reward_calibration.py # Calibration empirique (UK Road Safety & Zenodo) et métriques
├── models/
│   └── ppo_parking_default.zip # Modèle PPO pré-entraîné pour démonstration immédiate
├── app.py                    # Interface Web interactive Streamlit
└── README.md                 # Documentation et instructions de lancement
```

---

## Installation

Assurez-vous de disposer de **Python 3.10+**, puis installez les dépendances :

```bash
cd parkrl
pip install -r requirements.txt
```

---

## Lancement de l'interface Streamlit

```bash
cd parkrl
streamlit run app.py
```

L'application charge automatiquement le modèle pré-entraîné `models/ppo_parking_default.zip` s'il est présent, et vous permet également :

1. **Barre latérale :** ajuster les pénalités (`PENALTY_REVERSE`, `PENALTY_STEER_CHANGE`, `REWARD_DISTANCE_FACTOR`), lancer un nouvel entraînement PPO avec barre de progression et courbe de convergence, ou charger un modèle `.zip`.
2. **Simulateur Visuel & Raycasting :** observer l'agent pas à pas ou en animation continue avec ses **16 rayons laser (LiDAR)** colorés selon la distance, ses métriques en temps réel et la **carte de valeur Critique $V(x, y)$**.
3. **Comparaison IA vs Conducteurs Réels (Zenodo) :** superposer la trajectoire $(x, y)$ de l'agent PPO à des trajectoires humaines de référence et comparer durée, fluidité, marge de sécurité et ratio avant/arrière.
4. **Calibration Données Réelles :** recalculer les pénalités à partir de fichiers CSV **UK Road Safety** et **Zenodo**.
