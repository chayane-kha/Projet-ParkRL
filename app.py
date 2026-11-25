"""
app.py — Interface Web interactive Streamlit pour ParkRL.

Lancement :
    streamlit run app.py

Fonctionnalités principales :
1. Barre latérale : réglage des pénalités, entraînement PPO en direct avec barre de
   progression et courbe de convergence, chargement d'un modèle `.zip` pré-entraîné.
2. Simulateur Visuel & Raycasting en Temps Réel : exécution pas à pas ou animée,
   affichage des rayons LiDAR colorés, métriques de conduite et carte de valeur
   Critique (Value Function Heatmap 15x15).
3. Comparaison IA vs Conducteurs Réels (Zenodo) : superposition des trajectoires (x, y)
   et tableau comparatif des métriques clés.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import config
from agent.deep_rl_agent import DeepParkingAgent
from data_analysis.reward_calibration import (
    calibrate_penalties_from_data,
    compute_trajectory_metrics,
    generate_human_reference_trajectory,
    get_calibration_report,
)
from env.parking_env import ContinuousParkingEnv, create_oriented_car_polygon


st.set_page_config(
    page_title="ParkRL — Stationnement Automatique Deep RL",
    page_icon="🚗",
    layout="wide",
)


def _init_session_state() -> None:
    """Initialise l'environnement, l'agent PPO et l'état de simulation en session."""
    if "env" not in st.session_state:
        env = ContinuousParkingEnv()
        st.session_state.env = env
        agent = DeepParkingAgent(env)
        st.session_state.model_loaded_name = "Modèle non entraîné (initialisation aléatoire)"

        if config.DEFAULT_MODEL_PATH.exists():
            try:
                agent.load(config.DEFAULT_MODEL_PATH)
                st.session_state.model_loaded_name = config.DEFAULT_MODEL_PATH.name
            except Exception as exc:
                st.warning(f"Impossible de charger le modèle par défaut : {exc}")

        st.session_state.agent = agent
        obs, info = env.reset(seed=42)
        st.session_state.obs = obs
        st.session_state.info = info
        st.session_state.done = False
        st.session_state.reward_history = []


def _draw_parking_figure(env: ContinuousParkingEnv) -> plt.Figure:
    """Construit la figure Matplotlib haute lisibilité du simulateur et des rayons LiDAR."""
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
    ax.set_xlim(0.0, env.width)
    ax.set_ylim(0.0, env.height)
    ax.set_aspect("equal")
    ax.set_facecolor("#181d23")
    fig.patch.set_facecolor("#ffffff")

    # Marquages de l'allée
    ax.axhline(y=10.3, color="#6c757d", linestyle=":", linewidth=1.0, alpha=0.6)

    # Murs extérieurs
    for wall in env.walls:
        wx, wy = wall.exterior.xy
        ax.fill(wx, wy, color="#495057", zorder=2)

    # Place cible libre en vert
    tx, ty = env.target_box.exterior.xy
    ax.fill(tx, ty, color="#2ecc71", alpha=0.25, zorder=1)
    ax.plot(tx, ty, color="#27ae60", linewidth=2.0, zorder=2)
    ax.text(
        env.target_x,
        env.target_y + 2.55,
        "CIBLE",
        color="#2ecc71",
        fontsize=8,
        fontweight="bold",
        ha="center",
    )

    # Voitures déjà garées
    for pcar in env.parked_cars:
        px, py = pcar.exterior.xy
        ax.fill(px, py, color="#5c6bc0", alpha=0.9, zorder=3)
        ax.plot(px, py, color="#c5cae9", linewidth=1.2, zorder=3)

    # Trajectoire parcourue en pointillés
    if len(env.trajectory) > 1:
        traj_x = [pt["x"] for pt in env.trajectory]
        traj_y = [pt["y"] for pt in env.trajectory]
        ax.plot(traj_x, traj_y, "--", color="#f1c40f", linewidth=2.0, label="Trajectoire", zorder=4)

    # Rayons laser (LiDAR) colorés du vert (loin) au rouge (proche)
    for dist, (hx, hy) in env.last_rays:
        ratio = float(np.clip(dist / config.RAY_MAX_DIST, 0.0, 1.0))
        ray_color = plt.cm.RdYlGn(ratio)
        ax.plot([env.x, hx], [env.y, hy], color=ray_color, linewidth=1.5, alpha=0.9, zorder=5)
        if dist < config.RAY_MAX_DIST - 1e-2:
            ax.scatter([hx], [hy], color=ray_color, s=22, zorder=6)

    # Polygone orienté du véhicule
    car_poly = create_oriented_car_polygon(env.x, env.y, env.theta)
    cx, cy = car_poly.exterior.xy
    if env.status in ("Collision", "Immobilisé"):
        car_color = "#e74c3c"
    elif env.status == "Garé avec succès":
        car_color = "#2ecc71"
    else:
        car_color = "#3498db"

    car_patch = MplPolygon(
        list(zip(cx, cy)),
        closed=True,
        facecolor=car_color,
        edgecolor="#ffffff",
        linewidth=1.8,
        zorder=7,
    )
    ax.add_patch(car_patch)

    # Flèche indiquant l'avant du véhicule
    ax.arrow(
        env.x,
        env.y,
        1.45 * math.cos(env.theta),
        1.45 * math.sin(env.theta),
        head_width=0.35,
        head_length=0.35,
        fc="#ffffff",
        ec="#ffffff",
        zorder=8,
    )

    ax.set_title(
        f"Statut : {env.status} | t = {env.steps * env.dt:.1f}s | v = {env.v:+.2f} m/s",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("X (mètres)")
    ax.set_ylabel("Y (mètres)")
    ax.grid(color="#2c343e", linestyle="-", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


def _build_value_heatmap(agent: DeepParkingAgent, env: ContinuousParkingEnv) -> go.Figure:
    """Génère la carte de chaleur Plotly de la Value Function V(x, y) du réseau Critique."""
    xs, ys, value_grid = agent.evaluate_grid_values(env, grid_res=15)

    fig = go.Figure(
        data=go.Heatmap(
            z=value_grid,
            x=np.round(xs, 2),
            y=np.round(ys, 2),
            colorscale="Viridis",
            colorbar=dict(title="Valeur V(s)"),
            hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>V(s)=%{z:.2f}<extra></extra>",
        )
    )

    # Contour des voitures garées et de la place cible
    for cx in config.PARKED_CARS_X:
        fig.add_shape(
            type="rect",
            x0=cx - config.CAR_WIDTH / 2,
            y0=config.PARKING_ROW_Y - config.CAR_LENGTH / 2,
            x1=cx + config.CAR_WIDTH / 2,
            y1=config.PARKING_ROW_Y + config.CAR_LENGTH / 2,
            line=dict(color="white", width=1.5),
            fillcolor="rgba(255, 255, 255, 0.20)",
        )

    fig.add_shape(
        type="rect",
        x0=config.TARGET_X - config.SLOT_WIDTH / 2,
        y0=config.TARGET_Y - config.SLOT_LENGTH / 2,
        x1=config.TARGET_X + config.SLOT_WIDTH / 2,
        y1=config.TARGET_Y + config.SLOT_LENGTH / 2,
        line=dict(color="#2ecc71", width=2.5),
        fillcolor="rgba(46, 204, 113, 0.15)",
    )

    # Position actuelle du véhicule
    fig.add_trace(
        go.Scatter(
            x=[env.x],
            y=[env.y],
            mode="markers",
            marker=dict(size=12, color="#ff4b4b", symbol="diamond", line=dict(color="white", width=1.5)),
            name="Véhicule",
        )
    )

    fig.update_layout(
        title="Carte des valeurs du réseau Critique V(x, y)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        xaxis=dict(range=[0, env.width]),
        yaxis=dict(range=[0, env.height], scaleanchor="x", scaleratio=1),
        height=520,
        margin=dict(l=20, r=20, t=45, b=20),
        showlegend=False,
    )
    return fig


def _build_convergence_chart(reward_history: List[float]) -> go.Figure:
    """Trace la courbe d'évolution des récompenses d'épisodes pendant l'entraînement."""
    episodes = list(range(1, len(reward_history) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=reward_history,
            mode="lines",
            name="Récompense / épisode",
            line=dict(color="#5dade2", width=1.2),
            opacity=0.45,
        )
    )
    if len(reward_history) >= 5:
        window = min(20, len(reward_history))
        rolling = pd.Series(reward_history).rolling(window=window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=rolling,
                mode="lines",
                name=f"Moyenne mobile ({window})",
                line=dict(color="#e67e22", width=2.5),
            )
        )
    fig.update_layout(
        title="Convergence de l'entraînement PPO",
        xaxis_title="Épisode",
        yaxis_title="Récompense cumulée",
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _run_ppo_rollout(
    agent: DeepParkingAgent,
    env: ContinuousParkingEnv,
    start_pose: Dict[str, float],
) -> Tuple[List[Dict[str, float]], str]:
    """Exécute un épisode complet de l'agent PPO depuis une position de départ donnée."""
    sim_env = ContinuousParkingEnv(
        penalty_step=env.penalty_step,
        penalty_reverse=env.penalty_reverse,
        penalty_steer_change=env.penalty_steer_change,
        reward_distance_factor=env.reward_distance_factor,
        penalty_collision=env.penalty_collision,
        reward_parked=env.reward_parked,
    )
    obs, _ = sim_env.reset(options=start_pose)
    done = False
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = sim_env.step(action)
        done = terminated or truncated
    return list(sim_env.trajectory), sim_env.status


def _build_comparison_plot(
    env: ContinuousParkingEnv,
    ai_traj: List[Dict[str, float]],
    human_traj: List[Dict[str, float]],
) -> go.Figure:
    """Superpose les trajectoires (x, y) de l'agent PPO et du conducteur humain."""
    fig = go.Figure()

    # Voitures garées et place cible
    for idx, cx in enumerate(config.PARKED_CARS_X):
        fig.add_shape(
            type="rect",
            x0=cx - config.CAR_WIDTH / 2,
            y0=config.PARKING_ROW_Y - config.CAR_LENGTH / 2,
            x1=cx + config.CAR_WIDTH / 2,
            y1=config.PARKING_ROW_Y + config.CAR_LENGTH / 2,
            line=dict(color="#9fa8da", width=1.5),
            fillcolor="rgba(92, 107, 192, 0.75)",
        )

    fig.add_shape(
        type="rect",
        x0=config.TARGET_X - config.SLOT_WIDTH / 2,
        y0=config.TARGET_Y - config.SLOT_LENGTH / 2,
        x1=config.TARGET_X + config.SLOT_WIDTH / 2,
        y1=config.TARGET_Y + config.SLOT_LENGTH / 2,
        line=dict(color="#2ecc71", width=2.5),
        fillcolor="rgba(46, 204, 113, 0.20)",
    )

    # Trajectoire humaine (référence Zenodo)
    fig.add_trace(
        go.Scatter(
            x=[pt["x"] for pt in human_traj],
            y=[pt["y"] for pt in human_traj],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="#2ecc71", width=3, dash="dot"),
            name="Conducteur humain (référence Zenodo)",
        )
    )

    # Trajectoire IA (PPO)
    fig.add_trace(
        go.Scatter(
            x=[pt["x"] for pt in ai_traj],
            y=[pt["y"] for pt in ai_traj],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="#3498db", width=3),
            name="Agent Deep RL (PPO)",
        )
    )

    # Point de départ commun
    if human_traj:
        fig.add_trace(
            go.Scatter(
                x=[human_traj[0]["x"]],
                y=[human_traj[0]["y"]],
                mode="markers+text",
                marker=dict(size=12, color="#f39c12", symbol="star"),
                text=["Départ"],
                textposition="bottom center",
                name="Point de départ",
            )
        )

    fig.update_layout(
        title="Comparaison spatiale des trajectoires (x, y) — IA vs Conducteur Réel",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        xaxis=dict(range=[0, env.width]),
        yaxis=dict(range=[0, env.height], scaleanchor="x", scaleratio=1),
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _render_canvas_animation(
    env: ContinuousParkingEnv,
    frames: List[Dict[str, Any]],
    step_duration_ms: int = 120,
) -> None:
    """
    Affiche un lecteur HTML5 Canvas à 60 FPS avec interpolation fluide entre les pas
    de simulation, contrôles Lecture/Pause, Rejouer et barre de progression.
    """
    data = {
        "width": env.width,
        "height": env.height,
        "wall": config.WALL_THICKNESS,
        "carLength": config.CAR_LENGTH,
        "carWidth": config.CAR_WIDTH,
        "rayMax": config.RAY_MAX_DIST,
        "target": {
            "x": config.TARGET_X,
            "y": config.TARGET_Y,
            "w": config.SLOT_WIDTH,
            "h": config.SLOT_LENGTH,
        },
        "parkedCars": [
            {"x": cx, "y": config.PARKING_ROW_Y, "w": config.CAR_WIDTH, "h": config.CAR_LENGTH}
            for cx in config.PARKED_CARS_X
        ],
        "frames": frames,
        "stepDurationMs": step_duration_ms,
    }
    payload = json.dumps(data)

    html_code = f"""
    <div style="font-family: sans-serif; background: #ffffff; border-radius: 8px; padding: 8px;">
      <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        <button id="playBtn" style="padding:5px 12px; border-radius:6px; border:1px solid #ccc; cursor:pointer; background:#f0f2f6;">⏸️ Pause</button>
        <button id="replayBtn" style="padding:5px 12px; border-radius:6px; border:1px solid #ccc; cursor:pointer; background:#f0f2f6;">🔄 Rejouer</button>
        <input id="scrubber" type="range" min="0" max="{max(len(frames)-1, 0)}" step="0.02" value="0" style="flex:1;">
        <span id="hudText" style="font-size:13px; font-weight:600; min-width:210px; text-align:right;"></span>
      </div>
      <canvas id="parkCanvas" width="540" height="540" style="width:100%; max-width:540px; display:block; margin:auto; border-radius:8px; background:#181d23;"></canvas>
    </div>
    <script>
      const D = {payload};
      const canvas = document.getElementById("parkCanvas");
      const ctx = canvas.getContext("2d");
      const playBtn = document.getElementById("playBtn");
      const replayBtn = document.getElementById("replayBtn");
      const scrubber = document.getElementById("scrubber");
      const hudText = document.getElementById("hudText");

      const W = canvas.width, H = canvas.height;
      const sx = W / D.width, sy = H / D.height;
      const toX = (x) => x * sx;
      const toY = (y) => H - y * sy;

      function lerp(a, b, t) {{ return a + (b - a) * t; }}
      function lerpAngle(a, b, t) {{
        let d = (b - a + Math.PI) % (2 * Math.PI) - Math.PI;
        if (d < -Math.PI) d += 2 * Math.PI;
        return a + d * t;
      }}
      function rayColor(dist) {{
        const r = Math.max(0, Math.min(1, dist / D.rayMax));
        const red = Math.round(255 * (1 - r));
        const green = Math.round(220 * r + 35);
        return `rgb(${{red}},${{green}},60)`;
      }}

      let pos = 0.0;
      let playing = true;
      let lastTs = null;

      playBtn.onclick = () => {{
        playing = !playing;
        playBtn.textContent = playing ? "⏸️ Pause" : "▶️ Lecture";
        if (playing && pos >= D.frames.length - 1) pos = 0;
      }};
      replayBtn.onclick = () => {{
        pos = 0;
        playing = true;
        playBtn.textContent = "⏸️ Pause";
      }};
      scrubber.oninput = () => {{
        pos = parseFloat(scrubber.value);
        playing = false;
        playBtn.textContent = "▶️ Lecture";
      }};

      function draw(fIdx) {{
        const i0 = Math.min(Math.floor(fIdx), D.frames.length - 1);
        const i1 = Math.min(i0 + 1, D.frames.length - 1);
        const alpha = fIdx - i0;
        const f0 = D.frames[i0], f1 = D.frames[i1];

        const x = lerp(f0.x, f1.x, alpha);
        const y = lerp(f0.y, f1.y, alpha);
        const th = lerpAngle(f0.theta, f1.theta, alpha);
        const v = lerp(f0.v, f1.v, alpha);
        const t = lerp(f0.t, f1.t, alpha);
        const status = (i1 === D.frames.length - 1 && alpha > 0.85) ? f1.status : f0.status;

        ctx.clearRect(0, 0, W, H);

        // Grille légère
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 1;
        for (let g = 2; g < D.width; g += 2) {{
          ctx.beginPath(); ctx.moveTo(toX(g), 0); ctx.lineTo(toX(g), H); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(0, toY(g)); ctx.lineTo(W, toY(g)); ctx.stroke();
        }}

        // Murs extérieurs
        ctx.fillStyle = "#495057";
        const wt = D.wall * sx;
        ctx.fillRect(0, 0, W, wt);
        ctx.fillRect(0, H - wt, W, wt);
        ctx.fillRect(0, 0, wt, H);
        ctx.fillRect(W - wt, 0, wt, H);

        // Place cible verte
        const tg = D.target;
        ctx.fillStyle = "rgba(46, 204, 113, 0.25)";
        ctx.strokeStyle = "#27ae60";
        ctx.lineWidth = 2;
        ctx.fillRect(toX(tg.x - tg.w/2), toY(tg.y + tg.h/2), tg.w * sx, tg.h * sy);
        ctx.strokeRect(toX(tg.x - tg.w/2), toY(tg.y + tg.h/2), tg.w * sx, tg.h * sy);
        ctx.fillStyle = "#2ecc71";
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("CIBLE", toX(tg.x), toY(tg.y + tg.h/2 + 0.25));

        // Voitures garées
        D.parkedCars.forEach(c => {{
          ctx.fillStyle = "#5c6bc0";
          ctx.strokeStyle = "#c5cae9";
          ctx.lineWidth = 1.5;
          ctx.fillRect(toX(c.x - c.w/2), toY(c.y + c.h/2), c.w * sx, c.h * sy);
          ctx.strokeRect(toX(c.x - c.w/2), toY(c.y + c.h/2), c.w * sx, c.h * sy);
        }});

        // Trajectoire parcourue
        if (i0 >= 1) {{
          ctx.beginPath();
          ctx.setLineDash([6, 4]);
          ctx.strokeStyle = "#f1c40f";
          ctx.lineWidth = 2.2;
          ctx.moveTo(toX(D.frames[0].x), toY(D.frames[0].y));
          for (let k = 1; k <= i0; k++) ctx.lineTo(toX(D.frames[k].x), toY(D.frames[k].y));
          ctx.lineTo(toX(x), toY(y));
          ctx.stroke();
          ctx.setLineDash([]);
        }}

        // Rayons LiDAR
        const rays = f0.rays || [];
        rays.forEach(r => {{
          ctx.beginPath();
          ctx.strokeStyle = rayColor(r[0]);
          ctx.lineWidth = 1.5;
          ctx.moveTo(toX(x), toY(y));
          ctx.lineTo(toX(r[1]), toY(r[2]));
          ctx.stroke();
        }});

        // Véhicule orienté
        ctx.save();
        ctx.translate(toX(x), toY(y));
        ctx.rotate(-th);
        const cl = D.carLength * sx, cw = D.carWidth * sy;
        ctx.fillStyle = (status === "Collision" || status === "Immobilisé") ? "#e74c3c" : (status === "Garé avec succès" ? "#2ecc71" : "#3498db");
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.fillRect(-cl/2, -cw/2, cl, cw);
        ctx.strokeRect(-cl/2, -cw/2, cl, cw);
        // Flèche direction
        ctx.beginPath();
        ctx.moveTo(0, 0); ctx.lineTo(cl * 0.36, 0);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2.5; ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cl * 0.44, 0); ctx.lineTo(cl * 0.30, -5); ctx.lineTo(cl * 0.30, 5); ctx.closePath();
        ctx.fillStyle = "#ffffff"; ctx.fill();
        ctx.restore();

        hudText.textContent = `${{status}} | t=${{t.toFixed(1)}}s | v=${{v >= 0 ? '+' : ''}}${{v.toFixed(2)}} m/s`;
        scrubber.value = fIdx;
      }}

      function tick(ts) {{
        if (lastTs !== null && playing && D.frames.length > 1) {{
          const dtMs = ts - lastTs;
          pos += dtMs / D.stepDurationMs;
          if (pos >= D.frames.length - 1) {{
            pos = D.frames.length - 1;
            playing = false;
            playBtn.textContent = "▶️ Lecture";
          }}
        }}
        lastTs = ts;
        draw(pos);
        requestAnimationFrame(tick);
      }}
      requestAnimationFrame(tick);
    </script>
    """
    components.html(html_code, height=600)


def main() -> None:
    _init_session_state()
    env: ContinuousParkingEnv = st.session_state.env
    agent: DeepParkingAgent = st.session_state.agent

    st.title("🚗 ParkRL — Stationnement Automatique par Deep Reinforcement Learning")
    st.caption(
        "Agent **PPO (Stable-Baselines3)** dans un environnement continu **Gymnasium + Shapely** "
        "avec capteurs LiDAR, fonction de valeur Critique et calibration **UK Road Safety & Zenodo**."
    )

    # -----------------------------------------------------------------------
    # SECTION 1 : BARRE LATÉRALE (Hyperparamètres & Entraînement Deep RL)
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Hyperparamètres & Entraînement")
        st.info(f"**Modèle actif :** `{st.session_state.model_loaded_name}`")

        if not config.DEFAULT_MODEL_PATH.exists():
            st.warning(
                "Aucun modèle pré-entraîné trouvé sur disque. "
                "Cliquez sur **Entraîner le réseau PPO** (ex. 40 000 pas) pour démarrer."
            )

        st.subheader("Fonction de récompense")
        penalty_reverse = st.slider(
            "Pénalité marche arrière (PENALTY_REVERSE)",
            min_value=-1.00,
            max_value=0.00,
            value=float(env.penalty_reverse),
            step=0.05,
            help="Coût supplémentaire par pas en marche arrière (risque accru d'accidents).",
        )
        penalty_steer = st.slider(
            "Pénalité coup de volant (PENALTY_STEER_CHANGE)",
            min_value=-0.50,
            max_value=0.00,
            value=float(env.penalty_steer_change),
            step=0.02,
            help="Pénalise les variations brusques d'angle de braquage.",
        )
        reward_dist = st.slider(
            "Facteur rapprochement (REWARD_DISTANCE_FACTOR)",
            min_value=0.5,
            max_value=3.0,
            value=float(env.reward_distance_factor),
            step=0.1,
        )
        env.update_penalties(
            penalty_reverse=penalty_reverse,
            penalty_steer_change=penalty_steer,
            reward_distance_factor=reward_dist,
        )

        st.subheader("Entraînement PPO")
        total_timesteps = st.slider(
            "Nombre de pas d'entraînement (total_timesteps)",
            min_value=20_000,
            max_value=400_000,
            value=120_000,
            step=20_000,
        )

        if st.button("🚀 Entraîner le réseau PPO", use_container_width=True, type="primary"):
            progress_bar = st.progress(0, text="Initialisation de l'entraînement PPO...")
            chart_placeholder = st.empty()

            # Nouvel agent avec les pénalités courantes
            trained_agent = DeepParkingAgent(env)

            def _on_progress(step: int, total: int, rewards: List[float]) -> None:
                pct = min(1.0, step / max(total, 1))
                avg_r = np.mean(rewards[-15:]) if rewards else 0.0
                progress_bar.progress(
                    pct,
                    text=f"Pas {step:,} / {total:,} ({pct*100:.0f}%) — Récompense moy. : {avg_r:.1f}",
                )
                if len(rewards) >= 3:
                    chart_placeholder.plotly_chart(
                        _build_convergence_chart(rewards), use_container_width=True
                    )

            rewards = trained_agent.train(
                total_timesteps=total_timesteps,
                progress_callback=_on_progress,
            )
            saved_path = trained_agent.save(config.DEFAULT_MODEL_PATH)
            st.session_state.agent = trained_agent
            st.session_state.reward_history = rewards
            st.session_state.model_loaded_name = f"{saved_path.name} ({total_timesteps:,} pas)"
            agent = trained_agent

            obs, info = env.reset()
            st.session_state.obs = obs
            st.session_state.info = info
            st.session_state.done = False
            st.success(f"Entraînement terminé et sauvegardé dans `{saved_path.name}` !")

        if st.session_state.reward_history:
            st.plotly_chart(
                _build_convergence_chart(st.session_state.reward_history),
                use_container_width=True,
            )

        st.divider()
        st.subheader("💾 Chargement d'un modèle (.zip)")
        if config.DEFAULT_MODEL_PATH.exists():
            if st.button("Charger le modèle pré-entraîné par défaut", use_container_width=True):
                agent.load(config.DEFAULT_MODEL_PATH)
                st.session_state.model_loaded_name = config.DEFAULT_MODEL_PATH.name
                st.success("Modèle par défaut chargé !")

        uploaded_zip = st.file_uploader("Ou importer un fichier modèle `.zip`", type=["zip"])
        if uploaded_zip is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(uploaded_zip.read())
                tmp_path = tmp.name
            try:
                agent.load(tmp_path)
                st.session_state.model_loaded_name = uploaded_zip.name
                st.success(f"Modèle `{uploaded_zip.name}` chargé avec succès.")
            except Exception as exc:
                st.error(f"Erreur de chargement : {exc}")

    # -----------------------------------------------------------------------
    # ONGLETS PRINCIPAUX
    # -----------------------------------------------------------------------
    tab_sim, tab_cmp, tab_calib = st.tabs(
        [
            "🎯 Simulateur Visuel & Raycasting",
            "📊 Comparaison IA vs Conducteurs (Zenodo)",
            "📈 Calibration Données Réelles (UK & Zenodo)",
        ]
    )

    # -----------------------------------------------------------------------
    # SECTION 2 : SIMULATEUR VISUEL & RAYCASTING EN TEMPS RÉEL
    # -----------------------------------------------------------------------
    with tab_sim:
        col_btn1, col_btn2, col_btn3, col_speed = st.columns([1.2, 1.2, 1.8, 1.4])
        with col_btn1:
            btn_reset = st.button("🔄 Nouvel essai (Reset)", use_container_width=True)
        with col_btn2:
            btn_step = st.button(
                "⏭️ Pas suivant (Step)",
                use_container_width=True,
                disabled=st.session_state.done,
            )
        with col_btn3:
            btn_play = st.button("▶️ Jouer la manœuvre complète (Animation)", use_container_width=True)
        with col_speed:
            anim_delay = st.slider("Délai animation (s)", 0.02, 0.20, 0.10, 0.01)

        if btn_reset:
            obs, info = env.reset()
            st.session_state.obs = obs
            st.session_state.info = info
            st.session_state.done = False

        if btn_step and not st.session_state.done:
            action = agent.predict(st.session_state.obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(action)
            st.session_state.obs = obs
            st.session_state.info = info
            st.session_state.done = terminated or truncated

        # Emplacements dynamiques pour la mise à jour fluide pendant l'animation
        metrics_placeholder = st.empty()
        col_left, col_right = st.columns([1.05, 1.0])
        with col_left:
            sim_plot_placeholder = st.empty()
        with col_right:
            heatmap_placeholder = st.empty()

        def _render_metrics(info_dict: Dict[str, Any]) -> None:
            with metrics_placeholder.container():
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Temps écoulé", f"{info_dict['elapsed_time_s']:.1f} s")
                m2.metric("Dist. min. obstacles", f"{info_dict['min_obstacle_dist_cm']:.1f} cm")
                m3.metric("Temps marche arrière", f"{info_dict['reverse_time_s']:.1f} s")
                m4.metric("Récompense totale", f"{info_dict['total_reward']:.2f}")
                m5.metric("Statut", info_dict["status"])

        # Affichage de la heatmap Critique (calculée une fois par rendu)
        heatmap_fig = _build_value_heatmap(agent, env)
        heatmap_placeholder.plotly_chart(heatmap_fig, use_container_width=True)

        if btn_play:
            if st.session_state.done:
                obs, info = env.reset()
                st.session_state.obs = obs
                st.session_state.info = info
                st.session_state.done = False

            frames: List[Dict[str, Any]] = [
                {
                    "x": float(env.x),
                    "y": float(env.y),
                    "theta": float(env.theta),
                    "v": float(env.v),
                    "t": float(env.steps * env.dt),
                    "status": env.status,
                    "rays": [[round(d, 3), round(hx, 3), round(hy, 3)] for d, (hx, hy) in env.last_rays],
                }
            ]

            while not st.session_state.done:
                action = agent.predict(st.session_state.obs, deterministic=False)
                obs, _, terminated, truncated, info = env.step(action)
                st.session_state.obs = obs
                st.session_state.info = info
                st.session_state.done = terminated or truncated
                frames.append(
                    {
                        "x": float(env.x),
                        "y": float(env.y),
                        "theta": float(env.theta),
                        "v": float(env.v),
                        "t": float(env.steps * env.dt),
                        "status": env.status,
                        "rays": [[round(d, 3), round(hx, 3), round(hy, 3)] for d, (hx, hy) in env.last_rays],
                    }
                )

            _render_metrics(st.session_state.info)
            with sim_plot_placeholder.container():
                _render_canvas_animation(env, frames, step_duration_ms=int(max(30, anim_delay * 3000)))

            # Rafraîchir le marqueur véhicule sur la heatmap en fin d'animation
            heatmap_placeholder.plotly_chart(
                _build_value_heatmap(agent, env), use_container_width=True
            )
        else:
            _render_metrics(st.session_state.info)
            fig_sim = _draw_parking_figure(env)
            sim_plot_placeholder.pyplot(fig_sim, clear_figure=True)
            plt.close(fig_sim)

        # Détail des capteurs LiDAR
        with st.expander(f"📡 Valeurs instantanées des {config.NUM_RAYS} capteurs laser (LiDAR)"):
            ray_df = pd.DataFrame(
                [
                    {
                        "Capteur": f"Rayon {i + 1} ({int(math.degrees(env.ray_rel_angles[i]))}°)",
                        "Distance mesurée (m)": round(d, 2),
                        "Observation normalisée": round(d / config.RAY_MAX_DIST, 3),
                    }
                    for i, (d, _) in enumerate(env.last_rays)
                ]
            )
            st.dataframe(ray_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # SECTION 3 : COMPARAISON TRAJECTOIRES IA vs CONDUCTEURS RÉELS (ZENODO)
    # -----------------------------------------------------------------------
    with tab_cmp:
        st.subheader("Comparaison des trajectoires (x, y) : Agent PPO vs Conducteurs Humains")
        scenario = st.selectbox(
            "Trajectoire de référence humaine (profil Zenodo) :",
            [
                "Approche fluide par la gauche (Conducteur expérimenté)",
                "Approche fluide par la droite",
                "Manœuvre avec correction en marche arrière (créneau / reprise)",
            ],
        )

        human_traj = generate_human_reference_trajectory(env, scenario=scenario)
        start_pose = {
            "x": human_traj[0]["x"],
            "y": human_traj[0]["y"],
            "theta": human_traj[0]["theta"],
        }

        ai_traj, ai_status = _run_ppo_rollout(agent, env, start_pose=start_pose)

        col_plot, col_table = st.columns([1.3, 1.0])
        with col_plot:
            cmp_fig = _build_comparison_plot(env, ai_traj, human_traj)
            st.plotly_chart(cmp_fig, use_container_width=True)

        with col_table:
            st.markdown(f"**Statut final de l'agent PPO :** `{ai_status}`")
            ai_metrics = compute_trajectory_metrics(ai_traj)
            human_metrics = compute_trajectory_metrics(human_traj)

            comparison_df = pd.DataFrame(
                [
                    {
                        "Métrique": "Durée de manœuvre (s)",
                        "Agent PPO": ai_metrics["duration_s"],
                        "Conducteur Humain (Zenodo)": human_metrics["duration_s"],
                    },
                    {
                        "Métrique": "Fluidité du braquage (|Δδ| °/pas)",
                        "Agent PPO": ai_metrics["steer_smoothness_deg"],
                        "Conducteur Humain (Zenodo)": human_metrics["steer_smoothness_deg"],
                    },
                    {
                        "Métrique": "Marge de sécurité minimale (cm)",
                        "Agent PPO": ai_metrics["min_clearance_cm"],
                        "Conducteur Humain (Zenodo)": human_metrics["min_clearance_cm"],
                    },
                    {
                        "Métrique": "Ratio marche avant (%)",
                        "Agent PPO": ai_metrics["forward_ratio_pct"],
                        "Conducteur Humain (Zenodo)": human_metrics["forward_ratio_pct"],
                    },
                    {
                        "Métrique": "Ratio marche arrière (%)",
                        "Agent PPO": ai_metrics["reverse_ratio_pct"],
                        "Conducteur Humain (Zenodo)": human_metrics["reverse_ratio_pct"],
                    },
                    {
                        "Métrique": "Distance parcourue (m)",
                        "Agent PPO": ai_metrics["path_length_m"],
                        "Conducteur Humain (Zenodo)": human_metrics["path_length_m"],
                    },
                ]
            )
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.caption(
                "Les deux agents partent exactement de la même position initiale `(x, y, θ)` "
                "afin de comparer objectivement la fluidité, la durée et la marge de sécurité."
            )

    # -----------------------------------------------------------------------
    # SECTION BONUS : CALIBRATION EMPIRIQUE (UK ROAD SAFETY & ZENODO)
    # -----------------------------------------------------------------------
    with tab_calib:
        st.subheader("Calibration de la fonction de récompense depuis des données réelles")
        st.write(
            "Le module `data_analysis/reward_calibration.py` permet de recalculer les pénalités "
            "à partir de jeux de données CSV réels (**UK Road Safety DfT** et **Zenodo Parking Trajectories**)."
        )

        c1, c2 = st.columns(2)
        uk_file = c1.file_uploader("CSV UK Road Safety (colonnes `manoeuvre`, `accident_severity`)", type=["csv"])
        zen_file = c2.file_uploader("CSV Zenodo Trajectories (colonne `steer_rad`)", type=["csv"])

        uk_df = pd.read_csv(uk_file) if uk_file is not None else None
        zen_df = pd.read_csv(zen_file) if zen_file is not None else None

        report_df = get_calibration_report(uk_safety_df=uk_df, zenodo_traj_df=zen_df)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        calibrated_vals = calibrate_penalties_from_data(uk_safety_df=uk_df, zenodo_traj_df=zen_df)
        if st.button("Appliquer les pénalités calibrées à l'environnement"):
            env.update_penalties(
                penalty_reverse=calibrated_vals["PENALTY_REVERSE"],
                penalty_steer_change=calibrated_vals["PENALTY_STEER_CHANGE"],
            )
            st.success(
                f"Pénalités mises à jour : PENALTY_REVERSE={calibrated_vals['PENALTY_REVERSE']}, "
                f"PENALTY_STEER_CHANGE={calibrated_vals['PENALTY_STEER_CHANGE']}"
            )


if __name__ == "__main__":
    main()
