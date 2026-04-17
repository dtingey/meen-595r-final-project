# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib>=3.9",
#     "numpy>=2.1",
#     "pillow>=11.0",
# ]
# ///

"""Reproduction-oriented simulations for:

Li, P. & Duan, H. (2017)
"A potential game approach to multiple UAV cooperative search and surveillance"
Aerospace Science and Technology 68, 403-415.

This script implements the paper's core methodology in a self-contained way:
1) Potential-game-based coordinated motion with constrained actions.
2) Binary log-linear learning (BLLL) and best-response comparisons.
3) Cooperative search with Bayesian updates + consensus fusion.
4) Visualizations for state snapshots, convergence curves, trajectories, and GIFs.
5) An HTML report summarizing code changes, before/after, and impacts.

Notes on reproducibility:
- The paper text extraction was performed via `document_parse` and saved under
  `paper_extraction/` in this directory.
- A few simulation details in the paper (exact map equations / obstacle geometry)
  are not fully machine-readable from PDF equations, so this script reproduces
  the *methods and qualitative results* with transparent, documented assumptions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RectObstacle:
    """Axis-aligned rectangular obstacle in grid coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h


@dataclass
class CoverageRunResult:
    """Outputs from a coverage-control run."""

    potentials: np.ndarray
    positions_history: list[np.ndarray]
    active_history: list[np.ndarray]
    min_obstacle_distances: np.ndarray


@dataclass
class SearchRunResult:
    """Outputs from a cooperative search run."""

    uncertainty_curve: np.ndarray
    positions_history: list[np.ndarray]
    probability_snapshots: dict[int, np.ndarray]


# -----------------------------------------------------------------------------
# Potential-game coverage model (Section 3 of the paper)
# -----------------------------------------------------------------------------


class CoverageGame:
    """Coverage game with constrained actions and marginal-contribution utility.

    The game state is a set of UAV positions on a 2D grid. Utility for UAV i is
    implemented as marginal contribution:

        U_i(a_i, a_-i) = Φ(a_i, a_-i) - Φ(a_i^0, a_-i)

    where Φ is weighted coverage potential and a_i^0 means removing UAV i.

    In the implementation below, sensing quality is binary inside sensing radius
    (f=1 in range, 0 out of range), matching the indicator-style interpretation
    recoverable from the paper's equation extraction.
    """

    # 8-neighborhood + stay (max action count z_i^P = 9)
    ACTION_OFFSETS: tuple[tuple[int, int], ...] = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def __init__(
        self,
        density_map: np.ndarray,
        sensing_radii: np.ndarray,
        initial_positions: np.ndarray,
        obstacles_mask: np.ndarray | None = None,
    ) -> None:
        self.height, self.width = density_map.shape
        self.num_cells = self.width * self.height

        if obstacles_mask is None:
            obstacles_mask = np.zeros_like(density_map, dtype=bool)
        self.obstacles_mask = obstacles_mask.astype(bool)
        self.traversable_mask = ~self.obstacles_mask
        self.traversable_flat = self.traversable_mask.ravel()

        self.grid_x, self.grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.grid_x_flat = self.grid_x.ravel()
        self.grid_y_flat = self.grid_y.ravel()

        self.num_uavs = int(sensing_radii.size)
        self.sensing_radii = sensing_radii.astype(float)

        self.positions = initial_positions.astype(int).copy()
        if self.positions.shape != (self.num_uavs, 2):
            raise ValueError("initial_positions must have shape (num_uavs, 2)")

        # All UAVs active by default (failed UAVs can be deactivated later).
        self.active = np.ones(self.num_uavs, dtype=bool)

        # Internal coverage buffers.
        self.coverage_masks: list[np.ndarray] = [np.zeros(self.num_cells, dtype=bool) for _ in range(self.num_uavs)]
        self.coverage_counts = np.zeros(self.num_cells, dtype=np.int16)

        self.set_density_map(density_map)
        self._rebuild_coverage()

    def set_density_map(self, density_map: np.ndarray) -> None:
        """Update the global density map used in potential and utility.

        Important: we intentionally keep the original density magnitude (no
        normalization) so that utility values are on a realistic scale for the
        BLLL temperature parameter, consistent with the paper's formulation.
        """
        if density_map.shape != (self.height, self.width):
            raise ValueError("density_map shape mismatch")
        density = density_map.astype(float).copy()
        density[self.obstacles_mask] = 0.0
        if float(np.sum(density)) <= 0.0:
            density = self.traversable_mask.astype(float)
        self.density_map = density
        self.density_flat = density.ravel()

    def _coverage_mask_for_position(self, pos: tuple[int, int], radius: float) -> np.ndarray:
        """Boolean mask of cells covered by a UAV at position `pos`."""
        px, py = pos
        dx = self.grid_x_flat - px
        dy = self.grid_y_flat - py
        mask = (dx * dx + dy * dy) <= (radius * radius)
        return mask & self.traversable_flat

    def _rebuild_coverage(self) -> None:
        """Recompute all coverage masks and counts from current positions."""
        self.coverage_counts.fill(0)
        for i in range(self.num_uavs):
            if not self.active[i]:
                self.coverage_masks[i].fill(False)
                continue
            mask = self._coverage_mask_for_position(
                (int(self.positions[i, 0]), int(self.positions[i, 1])),
                float(self.sensing_radii[i]),
            )
            self.coverage_masks[i] = mask
            self.coverage_counts += mask.astype(np.int16)

    def deactivate_uavs(self, indices: list[int] | np.ndarray) -> None:
        """Deactivate UAVs (simulate failures), removing their coverage."""
        for idx in indices:
            i = int(idx)
            if i < 0 or i >= self.num_uavs:
                continue
            if not self.active[i]:
                continue
            self.coverage_counts -= self.coverage_masks[i].astype(np.int16)
            self.coverage_masks[i].fill(False)
            self.active[i] = False

    def constrained_actions(self, i: int) -> list[tuple[int, int]]:
        """Constrained action set C_{a_i}(t-1): local one-step motion + stay."""
        x, y = int(self.positions[i, 0]), int(self.positions[i, 1])
        actions: list[tuple[int, int]] = []
        for dx, dy in self.ACTION_OFFSETS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue
            if self.obstacles_mask[ny, nx]:
                continue
            actions.append((nx, ny))
        if not actions:
            actions = [(x, y)]
        return actions

    def _sample_trial_action_blll(self, i: int, rng: np.random.Generator) -> tuple[int, int]:
        """Sample trial action according to Eq. (7) style probabilities."""
        actions = self.constrained_actions(i)
        current = (int(self.positions[i, 0]), int(self.positions[i, 1]))

        if len(actions) == 1:
            return current

        z_i = 9.0  # maximum cardinality of constrained action set
        probs = []
        for action in actions:
            if action == current:
                p = 1.0 - (len(actions) - 1.0) / z_i
            else:
                p = 1.0 / z_i
            probs.append(max(0.0, p))

        p_arr = np.asarray(probs, dtype=float)
        p_arr /= float(np.sum(p_arr))

        pick = int(rng.choice(len(actions), p=p_arr))
        return actions[pick]

    def potential(self) -> float:
        """Potential Φ(a): weighted covered mass under current joint action."""
        return float(np.sum(self.density_flat[self.coverage_counts > 0]))

    def _utility_for_mask(self, mask_i: np.ndarray, others_covered: np.ndarray) -> float:
        """Marginal contribution utility for candidate coverage mask of UAV i."""
        unique_contribution = mask_i & (~others_covered)
        return float(np.sum(self.density_flat[unique_contribution]))

    def step_blll(self, temperature: float, rng: np.random.Generator) -> None:
        """One asynchronous Binary Log-Linear Learning update."""
        active_ids = np.flatnonzero(self.active)
        if active_ids.size == 0:
            return

        i = int(rng.choice(active_ids))  # choose random UAV
        current_pos = (int(self.positions[i, 0]), int(self.positions[i, 1]))
        trial_pos = self._sample_trial_action_blll(i, rng)  # sample trial action

        current_mask = self.coverage_masks[i]
        others_covered = (self.coverage_counts - current_mask.astype(np.int16)) > 0

        u_current = self._utility_for_mask(current_mask, others_covered)  # calculate current utility
        if trial_pos == current_pos:
            trial_mask = current_mask
            u_trial = u_current
        else:
            trial_mask = self._coverage_mask_for_position(trial_pos, self.sensing_radii[i])
            u_trial = self._utility_for_mask(trial_mask, others_covered)  # calculate trial utility

        if temperature <= 0.0:
            choose_trial = u_trial > u_current
        else:
            # softmax over the two utilities.
            u_max = max(u_current, u_trial)
            e_cur = math.exp((u_current - u_max) / temperature)
            e_tri = math.exp((u_trial - u_max) / temperature)
            p_trial = e_tri / (e_tri + e_cur)
            choose_trial = bool(rng.random() < p_trial)

        if choose_trial and trial_pos != current_pos:
            delta = trial_mask.astype(np.int16) - current_mask.astype(np.int16)
            self.coverage_counts += delta
            self.coverage_masks[i] = trial_mask
            self.positions[i, 0] = trial_pos[0]
            self.positions[i, 1] = trial_pos[1]

    def step_best_response(self, rng: np.random.Generator) -> None:
        """One asynchronous best-response update over constrained actions."""
        active_ids = np.flatnonzero(self.active)
        if active_ids.size == 0:
            return

        i = int(rng.choice(active_ids))
        current_pos = (int(self.positions[i, 0]), int(self.positions[i, 1]))
        current_mask = self.coverage_masks[i]
        others_covered = (self.coverage_counts - current_mask.astype(np.int16)) > 0

        actions = self.constrained_actions(i)
        utilities = np.empty(len(actions), dtype=float)
        masks: list[np.ndarray] = []
        for k, action in enumerate(actions):
            if action == current_pos:
                mask = current_mask
            else:
                mask = self._coverage_mask_for_position(action, self.sensing_radii[i])
            masks.append(mask)
            utilities[k] = self._utility_for_mask(mask, others_covered)

        best_u = float(np.max(utilities))
        best_ids = np.flatnonzero(np.isclose(utilities, best_u, atol=1e-12))
        pick = int(rng.choice(best_ids))
        best_action = actions[pick]

        if best_action != current_pos:
            best_mask = masks[pick]
            delta = best_mask.astype(np.int16) - current_mask.astype(np.int16)
            self.coverage_counts += delta
            self.coverage_masks[i] = best_mask
            self.positions[i, 0] = best_action[0]
            self.positions[i, 1] = best_action[1]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def time_axis(
    num_points: int,
    *,
    num_uavs: int,
    updates_per_step: int = 1,
) -> np.ndarray:
    """Return x-axis values in paper-style timesteps.

    One timestep is treated as a network-sweep equivalent:
    logical_step * (updates_per_step / num_uavs).
    """
    x = np.arange(num_points, dtype=float)
    if num_uavs > 0:
        x = x * float(updates_per_step) / float(num_uavs)
    return x


def time_value(
    step: float,
    *,
    num_uavs: int,
    updates_per_step: int = 1,
) -> float:
    """Convert a single step value to paper-style timestep units."""
    if num_uavs > 0:
        return float(step) * float(updates_per_step) / float(num_uavs)
    return float(step)


def time_xlabel(*, domain: str = "learning") -> str:
    """Consistent x-axis label (paper-style timestep)."""
    return f"{domain} timestep (network-sweep equivalent)"


def left_edge_initial_positions(
    num_uavs: int,
    width: int,
    height: int,
    x: int = 4,
    pad: int = 3,
    obstacles_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Place UAVs roughly uniformly along the left edge, avoiding obstacles."""
    ys = np.linspace(pad, height - 1 - pad, num_uavs).round().astype(int)
    xs = np.full_like(ys, x)
    pos = np.column_stack((xs, ys))

    if obstacles_mask is None:
        return pos

    # If any start point collides with an obstacle, shift right until free.
    for i in range(num_uavs):
        px, py = int(pos[i, 0]), int(pos[i, 1])
        while px < width and obstacles_mask[py, px]:
            px += 1
        pos[i, 0] = min(px, width - 1)
    return pos


def make_prior_density_map(width: int, height: int) -> np.ndarray:
    """Construct a smooth prior density map (sum of anisotropic Gaussians)."""
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    components = [
        (1.00, 0.20 * width, 0.25 * height, 0.10 * width, 0.14 * height),
        (1.15, 0.68 * width, 0.34 * height, 0.14 * width, 0.12 * height),
        (0.90, 0.52 * width, 0.76 * height, 0.10 * width, 0.15 * height),
        (0.65, 0.83 * width, 0.80 * height, 0.09 * width, 0.10 * height),
    ]

    rho = np.zeros((height, width), dtype=float)
    for amp, cx, cy, sx, sy in components:
        dx2 = ((xx - cx) ** 2) / (2.0 * sx * sx)
        dy2 = ((yy - cy) ** 2) / (2.0 * sy * sy)
        rho += amp * np.exp(-(dx2 + dy2))

    rho += 0.02
    return rho


def make_target_probability_map(width: int, height: int) -> np.ndarray:
    """Target-presence probability map for cooperative search experiments.

    We intentionally keep the target field sparse so the posterior map should
    converge to near-zero over most cells with only a small set of high-
    probability target cells.
    """
    base = make_prior_density_map(width, height)
    normalized = base / np.max(base)
    # Sparse Bernoulli field with hotspot structure.
    target_prob = 0.001 + 0.02 * normalized
    target_prob = np.clip(target_prob, 0.0005, 0.03)
    return target_prob


def make_obstacle_mask(
    width: int,
    height: int,
    obstacles: list[RectObstacle],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for obs in obstacles:
        x0 = max(0, obs.x)
        y0 = max(0, obs.y)
        x1 = min(width, obs.x1)
        y1 = min(height, obs.y1)
        mask[y0:y1, x0:x1] = True
    return mask


def make_uniform_density(width: int, height: int, obstacles_mask: np.ndarray | None = None) -> np.ndarray:
    density = np.ones((height, width), dtype=float)
    if obstacles_mask is not None:
        density[obstacles_mask] = 0.0
    return density


def min_distance_to_rectangles(positions: np.ndarray, obstacles: list[RectObstacle]) -> float:
    """Minimum Euclidean distance from any UAV to any rectangular obstacle."""
    if not obstacles:
        return float("nan")

    min_d = float("inf")
    for x, y in positions:
        px, py = float(x), float(y)
        for obs in obstacles:
            dx = max(obs.x - px, 0.0, px - obs.x1)
            dy = max(obs.y - py, 0.0, py - obs.y1)
            d = math.hypot(dx, dy)
            min_d = min(min_d, d)
    return min_d


def run_coverage_game(
    game: CoverageGame,
    steps: int,
    algorithm: str,
    rng: np.random.Generator,
    temperature: float = 0.2,
    record_positions: bool = True,
    record_every: int = 1,
    failures: dict[int, list[int]] | None = None,
    obstacle_list: list[RectObstacle] | None = None,
    updates_per_step: int = 1,
) -> CoverageRunResult:
    """Run coverage dynamics and collect time-series outputs.

    `steps` are logical simulation steps. Each logical step executes
    `updates_per_step` asynchronous UAV updates.
    """
    if failures is None:
        failures = {}
    if obstacle_list is None:
        obstacle_list = []
    if updates_per_step < 1:
        raise ValueError("updates_per_step must be >= 1")

    potentials = np.zeros(steps + 1, dtype=float)
    min_dist = np.zeros(steps + 1, dtype=float)

    positions_history: list[np.ndarray] = []
    active_history: list[np.ndarray] = []

    potentials[0] = game.potential()
    min_dist[0] = min_distance_to_rectangles(game.positions, obstacle_list)

    if record_positions:
        positions_history.append(game.positions.copy())
        active_history.append(game.active.copy())

    for t in range(1, steps + 1):
        if t in failures:
            game.deactivate_uavs(failures[t])

        for _ in range(updates_per_step):
            if algorithm == "blll":
                game.step_blll(temperature=temperature, rng=rng)
            elif algorithm == "best_response":
                game.step_best_response(rng=rng)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        potentials[t] = game.potential()
        min_dist[t] = min_distance_to_rectangles(game.positions, obstacle_list)

        if record_positions and (t % record_every == 0):
            positions_history.append(game.positions.copy())
            active_history.append(game.active.copy())

    return CoverageRunResult(
        potentials=potentials,
        positions_history=positions_history,
        active_history=active_history,
        min_obstacle_distances=min_dist,
    )


# -----------------------------------------------------------------------------
# Cooperative search model (Section 4 and 5.2 of paper)
# -----------------------------------------------------------------------------


def metropolis_weights(positions: np.ndarray, comm_range: float) -> np.ndarray:
    """Metropolis weight matrix over communication graph."""
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    pos = positions.astype(float)
    dx = pos[:, None, 0] - pos[None, :, 0]
    dy = pos[:, None, 1] - pos[None, :, 1]
    dist2 = dx * dx + dy * dy
    adjacency = dist2 <= (comm_range * comm_range)

    # Exclude diagonal for degree, as in standard graph degree definition.
    np.fill_diagonal(adjacency, False)
    degrees = np.sum(adjacency, axis=1)

    w = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                wij = 1.0 / (1.0 + max(degrees[i], degrees[j]))
                w[i, j] = wij
                w[j, i] = wij

    diag = 1.0 - np.sum(w, axis=1)
    w[np.arange(n), np.arange(n)] = diag
    return w


def run_cooperative_search(
    *,
    width: int,
    height: int,
    num_uavs: int,
    sensing_radii: np.ndarray,
    comm_range: float,
    detection_prob: float,
    false_alarm_prob: float,
    sampling_frequency: int,
    steps: int,
    temperature: float,
    uncertainty_gain: float,
    rng: np.random.Generator,
    snapshot_steps: tuple[int, ...] = (),
    obstacles_mask: np.ndarray | None = None,
    record_positions: bool = True,
    record_every: int = 1,
    updates_per_step: int = 1,
) -> SearchRunResult:
    """Run cooperative search with Bayes update + consensus fusion.

    Motion control uses BLLL against a dynamic uncertainty density map.
    Information update uses:
      H_{i,g,t} = H_{i,g,t-1} + log(pc/pf) when z=1
      H_{i,g,t} = H_{i,g,t-1} + log((1-pc)/(1-pf)) when z=0
    followed by consensus fusion Q_t = W_t H_t.

    Uncertainty map used for motion:
      φ_{i,g,t} = exp(-k * |Q_{i,g,t}|)
    which matches the paper's qualitative behavior (initial avg uncertainty 1,
    convergence toward 0 as confidence grows).
    """
    if obstacles_mask is None:
        obstacles_mask = np.zeros((height, width), dtype=bool)
    if updates_per_step < 1:
        raise ValueError("updates_per_step must be >= 1")

    target_prob = make_target_probability_map(width, height)
    target_map = rng.random((height, width)) < target_prob
    target_flat = target_map.ravel()

    init_positions = left_edge_initial_positions(
        num_uavs=num_uavs,
        width=width,
        height=height,
        x=4,
        pad=3,
        obstacles_mask=obstacles_mask,
    )

    # Start with uniform uncertainty density (all cells equally unknown).
    initial_density = make_uniform_density(width, height, obstacles_mask=obstacles_mask)
    game = CoverageGame(
        density_map=initial_density,
        sensing_radii=sensing_radii,
        initial_positions=init_positions,
        obstacles_mask=obstacles_mask,
    )

    m = width * height
    h_map = np.zeros((num_uavs, m), dtype=float)  # H in paper
    q_map = np.zeros((num_uavs, m), dtype=float)  # Q in paper

    # Constants for Bayesian update in transformed domain.
    # We store H = log((1 - P) / P), so additive updates use NEGATIVE
    # log-likelihood ratios. This makes non-target cells trend to P≈0 and
    # true-target cells trend to P≈1.
    log_hit = -math.log(detection_prob / false_alarm_prob)
    log_miss = -math.log((1.0 - detection_prob) / (1.0 - false_alarm_prob))

    uncertainty_curve = np.zeros(steps + 1, dtype=float)
    uncertainty_curve[0] = 1.0  # because q_map starts at zeros

    probability_snapshots: dict[int, np.ndarray] = {0: np.full((height, width), 0.5)}
    positions_history: list[np.ndarray] = [game.positions.copy()] if record_positions else []

    # MAIN LOOP
    for t in range(1, steps + 1):
        for _ in range(updates_per_step):
            game.step_blll(temperature=temperature, rng=rng)

        # Sensor observations + information fusion every FS logical steps.
        if t % sampling_frequency == 0:
            for i in range(num_uavs):
                if not game.active[i]:
                    continue

                covered = game.coverage_masks[i]
                idx = np.flatnonzero(covered)
                if idx.size == 0:
                    continue

                true_targets = target_flat[idx]
                randv = rng.random(idx.size)
                # Z=1 with pc if target exists, else with pf.
                detections = np.where(true_targets, randv < detection_prob, randv < false_alarm_prob)

                # Eq. (26)-style additive update in transformed domain H.
                h_map[i, idx] += np.where(detections, log_hit, log_miss)

            # Consensus fusion using Metropolis weights (Eq. 28 + Eq. 29).
            w = metropolis_weights(game.positions, comm_range=comm_range)
            q_map = w @ h_map
            h_map = q_map.copy()

        # Build uncertainty density for next motion decisions.
        phi_i = np.exp(-uncertainty_gain * np.abs(q_map))
        uncertainty_curve[t] = float(np.mean(phi_i))

        density = np.mean(phi_i, axis=0).reshape(height, width)
        density[obstacles_mask] = 0.0
        if float(np.sum(density)) <= 0.0:
            density = make_uniform_density(width, height, obstacles_mask=obstacles_mask)

        game.set_density_map(density)

        if t in snapshot_steps:
            # Inverse transform of H = ln(1/P - 1) => P = 1/(1+exp(H))
            h0 = np.clip(h_map[0], -50.0, 50.0)
            p0 = 1.0 / (1.0 + np.exp(h0))
            probability_snapshots[t] = p0.reshape(height, width)

        if record_positions and (t % record_every == 0):
            positions_history.append(game.positions.copy())

    return SearchRunResult(
        uncertainty_curve=uncertainty_curve,
        positions_history=positions_history,
        probability_snapshots=probability_snapshots,
    )


# -----------------------------------------------------------------------------
# Plotting / visualization helpers
# -----------------------------------------------------------------------------


def plot_state(
    ax: plt.Axes,
    density_map: np.ndarray,
    positions: np.ndarray,
    sensing_radii: np.ndarray,
    *,
    active: np.ndarray | None = None,
    failed_indices: list[int] | None = None,
    obstacles: list[RectObstacle] | None = None,
    title: str,
) -> None:
    """Plot a map, UAV locations, and sensing circles."""
    if active is None:
        active = np.ones(positions.shape[0], dtype=bool)
    if failed_indices is None:
        failed_indices = []
    if obstacles is None:
        obstacles = []

    ax.imshow(density_map, origin="lower", cmap="viridis", alpha=0.92)

    for obs in obstacles:
        ax.add_patch(
            Rectangle(
                (obs.x, obs.y),
                obs.w,
                obs.h,
                edgecolor="black",
                facecolor="black",
                alpha=0.35,
            )
        )

    # Plot active UAVs and sensing circles.
    for i, (x, y) in enumerate(positions):
        if active[i]:
            ax.scatter(x, y, c="white", s=28, edgecolors="black", linewidths=0.5, zorder=3)
            circ = Circle(
                (x, y),
                radius=float(sensing_radii[i]),
                edgecolor="white",
                facecolor="none",
                lw=0.6,
                alpha=0.6,
                zorder=2,
            )
            ax.add_patch(circ)

    # Highlight explicitly failed UAVs.
    for i in failed_indices:
        x, y = positions[i]
        ax.scatter(x, y, c="red", s=56, marker="x", linewidths=2.0, zorder=4)

    ax.set_title(title)
    ax.set_xlim(-0.5, density_map.shape[1] - 0.5)
    ax.set_ylim(-0.5, density_map.shape[0] - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_trajectories(
    out_path: Path,
    density_map: np.ndarray,
    positions_history: list[np.ndarray],
    *,
    obstacles: list[RectObstacle] | None = None,
    title: str,
) -> None:
    """Save static trajectories (agent paths over time)."""
    if not positions_history:
        return
    if obstacles is None:
        obstacles = []

    arr = np.stack(positions_history, axis=0)  # (T, N, 2)
    t_steps, n_uavs = arr.shape[0], arr.shape[1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(density_map, origin="lower", cmap="viridis", alpha=0.35)

    for obs in obstacles:
        ax.add_patch(Rectangle((obs.x, obs.y), obs.w, obs.h, edgecolor="black", facecolor="black", alpha=0.25))

    colors = plt.cm.tab20(np.linspace(0, 1, max(n_uavs, 2)))
    for i in range(n_uavs):
        ax.plot(arr[:, i, 0], arr[:, i, 1], color=colors[i % len(colors)], lw=1.1, alpha=0.85)
        ax.scatter(arr[0, i, 0], arr[0, i, 1], color=colors[i % len(colors)], s=18, marker="o")
        ax.scatter(arr[-1, i, 0], arr[-1, i, 1], color=colors[i % len(colors)], s=24, marker="s")

    ax.set_title(f"{title}\n(start=o, end=s, frames={t_steps})")
    ax.set_xlim(-0.5, density_map.shape[1] - 0.5)
    ax.set_ylim(-0.5, density_map.shape[0] - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_movement_gif(
    out_path: Path,
    density_map: np.ndarray,
    positions_history: list[np.ndarray],
    *,
    obstacles: list[RectObstacle] | None = None,
    title: str,
    frame_stride: int = 6,
    fps: int = 10,
) -> None:
    """Create an animated GIF showing agent movement over time."""
    if not positions_history:
        return
    if obstacles is None:
        obstacles = []

    frames = positions_history[:: max(1, frame_stride)]
    if len(frames) < 2:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(density_map, origin="lower", cmap="viridis", alpha=0.45)
    for obs in obstacles:
        ax.add_patch(Rectangle((obs.x, obs.y), obs.w, obs.h, edgecolor="black", facecolor="black", alpha=0.25))

    scat = ax.scatter(frames[0][:, 0], frames[0][:, 1], c="white", s=28, edgecolors="black")
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", color="white", fontsize=10)

    ax.set_xlim(-0.5, density_map.shape[1] - 0.5)
    ax.set_ylim(-0.5, density_map.shape[0] - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def _update(frame_idx: int):
        pos = frames[frame_idx]
        scat.set_offsets(pos)
        txt.set_text(f"frame {frame_idx + 1}/{len(frames)}")
        return scat, txt

    anim = FuncAnimation(fig, _update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def make_before_placeholder(path: Path) -> None:
    """A tiny 'before' visualization for report comparison."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        'Before changes: sim.py only printed "Hello from sim.py!"\n(no simulations, no figures, no analysis)',
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Scenario implementations (paper Sections 5.1 and 5.2)
# -----------------------------------------------------------------------------


def mode_value(
    *,
    quick: bool,
    quick_value: Any,
    default_value: Any,
) -> Any:
    """Pick a parameter value based on run mode (quick vs default)."""
    if quick:
        return quick_value
    return default_value


def scenario_homogeneous_with_failures(
    output_dir: Path,
    rng: np.random.Generator,
    quick: bool,
) -> dict[str, Any]:
    """Reproduce Fig. 4 style homogeneous coverage + failure reconfiguration."""
    width, height = 100, 100
    n = 15
    radius = 10.0
    updates_per_step = n

    # Convergence-focused horizons by mode.
    steps_stage1 = int(
        mode_value(
            quick=quick,
            quick_value=250,
            default_value=400,
        )
    )
    steps_stage2 = int(
        mode_value(
            quick=quick,
            quick_value=250,
            default_value=2100,
        )
    )
    temperature = 0.2

    density = make_prior_density_map(width, height)
    init_positions = left_edge_initial_positions(n, width, height, x=4, pad=3)
    sensing_radii = np.full(n, radius, dtype=float)

    game = CoverageGame(
        density_map=density,
        sensing_radii=sensing_radii,
        initial_positions=init_positions,
    )

    run1 = run_coverage_game(
        game=game,
        steps=steps_stage1,
        algorithm="blll",
        rng=rng,
        temperature=temperature,
        record_positions=True,
        record_every=1,
        updates_per_step=updates_per_step,
    )

    snapshot_after_stage1 = game.positions.copy()
    failed_ids = [4, 11]
    game.deactivate_uavs(failed_ids)

    run2 = run_coverage_game(
        game=game,
        steps=steps_stage2,
        algorithm="blll",
        rng=rng,
        temperature=temperature,
        record_positions=True,
        record_every=1,
        updates_per_step=updates_per_step,
    )

    potentials = np.concatenate([run1.potentials, run2.potentials])

    # Combine movement histories.
    positions_history = run1.positions_history + run2.positions_history
    active_history = run1.active_history + run2.active_history

    fig_path = output_dir / "fig4_homogeneous_with_failures.png"
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))

    plot_state(
        axs[0, 0],
        density,
        init_positions,
        sensing_radii,
        title="(a) Initial positions (homogeneous UAVs)",
    )
    stage1_time_display = time_value(
        steps_stage1,
        num_uavs=n,
        updates_per_step=updates_per_step,
    )
    plot_state(
        axs[0, 1],
        density,
        snapshot_after_stage1,
        sensing_radii,
        title=(f"(b) After {stage1_time_display:.1f} BLLL timesteps (failed UAVs highlighted)"),
        failed_indices=failed_ids,
    )
    plot_state(
        axs[1, 0],
        density,
        game.positions,
        sensing_radii,
        active=game.active,
        title="(c) Reconfigured final state after failures",
    )

    x_time = time_axis(
        len(potentials),
        num_uavs=n,
        updates_per_step=updates_per_step,
    )
    x_fail = time_value(
        steps_stage1,
        num_uavs=n,
        updates_per_step=updates_per_step,
    )
    axs[1, 1].plot(x_time, potentials, lw=1.6, color="tab:blue")
    axs[1, 1].axvline(x_fail, color="tab:red", ls="--", lw=1.1, label="failure injected")
    axs[1, 1].set_title("(d) Potential evolution")
    axs[1, 1].set_xlabel(time_xlabel(domain="learning"))
    axs[1, 1].set_ylabel("potential Φ(a)")
    axs[1, 1].grid(alpha=0.3)
    axs[1, 1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    traj_path = output_dir / "homogeneous_trajectories.png"
    plot_trajectories(
        traj_path,
        density,
        positions_history,
        title="Homogeneous coverage: UAV trajectories over time",
    )

    gif_path = output_dir / "homogeneous_movement.gif"
    save_movement_gif(
        gif_path,
        density,
        positions_history,
        title="Homogeneous coverage movement (BLLL)",
        frame_stride=int(
            mode_value(
                quick=quick,
                quick_value=8,
                default_value=5,
            )
        ),
        fps=10,
    )

    return {
        "density": density,
        "init_positions": init_positions,
        "sensing_radii": sensing_radii,
        "fig_path": fig_path,
        "traj_path": traj_path,
        "gif_path": gif_path,
        "final_potential": float(potentials[-1]),
        "potentials": potentials,
        "positions_history": positions_history,
        "active_history": active_history,
    }


def scenario_heterogeneous_coverage(
    output_dir: Path,
    rng: np.random.Generator,
    quick: bool,
) -> dict[str, Any]:
    """Reproduce Fig. 5 style heterogeneous sensing-range coverage."""
    width, height = 100, 100
    n = 15
    updates_per_step = n
    # Convergence-focused horizons by mode.
    steps = int(
        mode_value(
            quick=quick,
            quick_value=300,
            default_value=2500,
        )
    )
    temperature = 0.2

    density = make_prior_density_map(width, height)
    init_positions = left_edge_initial_positions(n, width, height, x=4, pad=3)

    sensing_radii = np.concatenate(
        [
            np.full(5, 12.0),
            np.full(5, 10.0),
            np.full(5, 8.0),
        ]
    )

    game = CoverageGame(
        density_map=density,
        sensing_radii=sensing_radii,
        initial_positions=init_positions,
    )

    run = run_coverage_game(
        game=game,
        steps=steps,
        algorithm="blll",
        rng=rng,
        temperature=temperature,
        record_positions=False,
        updates_per_step=updates_per_step,
    )

    fig_path = output_dir / "fig5_heterogeneous_coverage.png"
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))

    plot_state(
        axs[0],
        density,
        game.positions,
        sensing_radii,
        title="(a) Final configuration (heterogeneous sensing radii)",
    )

    x_time = time_axis(
        len(run.potentials),
        num_uavs=n,
        updates_per_step=updates_per_step,
    )
    axs[1].plot(x_time, run.potentials, lw=1.6, color="tab:green")
    axs[1].set_title("(b) Potential evolution")
    axs[1].set_xlabel(time_xlabel(domain="learning"))
    axs[1].set_ylabel("potential Φ(a)")
    axs[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return {
        "fig_path": fig_path,
        "final_potential": float(run.potentials[-1]),
        "potentials": run.potentials,
    }


def scenario_obstacle_coverage(
    output_dir: Path,
    rng: np.random.Generator,
    quick: bool,
) -> dict[str, Any]:
    """Reproduce Fig. 6 style coverage with non-convex obstacles."""
    width, height = 100, 80
    n = 15
    # Per request: obstacle experiment uses sensing radius 12 (area remains 100x80).
    radius = 12.0
    updates_per_step = n
    # Convergence-focused horizons by mode.
    steps = int(
        mode_value(
            quick=quick,
            quick_value=350,
            default_value=2500,
        )
    )
    temperature = 0.2

    obstacles = [
        RectObstacle(18, 14, 14, 20),
        RectObstacle(43, 8, 16, 24),
        RectObstacle(40, 44, 20, 14),
        RectObstacle(73, 31, 14, 28),
    ]
    obstacles_mask = make_obstacle_mask(width, height, obstacles)

    density = make_uniform_density(width, height, obstacles_mask=obstacles_mask)
    init_positions = left_edge_initial_positions(n, width, height, x=4, pad=3, obstacles_mask=obstacles_mask)
    sensing_radii = np.full(n, radius, dtype=float)

    game = CoverageGame(
        density_map=density,
        sensing_radii=sensing_radii,
        initial_positions=init_positions,
        obstacles_mask=obstacles_mask,
    )

    run = run_coverage_game(
        game=game,
        steps=steps,
        algorithm="blll",
        rng=rng,
        temperature=temperature,
        record_positions=True,
        record_every=1,
        obstacle_list=obstacles,
        updates_per_step=updates_per_step,
    )

    fig_path = output_dir / "fig6_obstacle_coverage.png"
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))

    plot_state(
        axs[0, 0],
        density,
        init_positions,
        sensing_radii,
        obstacles=obstacles,
        title="(a) Initial positions with obstacles",
    )
    plot_state(
        axs[0, 1],
        density,
        game.positions,
        sensing_radii,
        obstacles=obstacles,
        title="(b) Final configuration after cooperative motion",
    )

    x_time = time_axis(
        len(run.potentials),
        num_uavs=n,
        updates_per_step=updates_per_step,
    )
    axs[1, 0].plot(x_time, run.potentials, lw=1.6, color="tab:purple")
    axs[1, 0].set_title("(c) Potential evolution")
    axs[1, 0].set_xlabel(time_xlabel(domain="learning"))
    axs[1, 0].set_ylabel("potential Φ(a)")
    axs[1, 0].grid(alpha=0.3)

    axs[1, 1].plot(x_time, run.min_obstacle_distances, lw=1.4, color="tab:orange")
    axs[1, 1].set_title("(d) Minimum distance to obstacles")
    axs[1, 1].set_xlabel(time_xlabel(domain="learning"))
    axs[1, 1].set_ylabel("minimum distance (cells)")
    axs[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    traj_path = output_dir / "obstacle_trajectories.png"
    plot_trajectories(
        traj_path,
        density,
        run.positions_history,
        obstacles=obstacles,
        title="Coverage with obstacles: UAV trajectories",
    )

    gif_path = output_dir / "obstacle_movement.gif"
    save_movement_gif(
        gif_path,
        density,
        run.positions_history,
        obstacles=obstacles,
        title="Obstacle scenario movement (BLLL)",
        frame_stride=int(
            mode_value(
                quick=quick,
                quick_value=8,
                default_value=5,
            )
        ),
        fps=10,
    )

    return {
        "fig_path": fig_path,
        "traj_path": traj_path,
        "gif_path": gif_path,
        "final_potential": float(run.potentials[-1]),
        "potentials": run.potentials,
        "obstacles": obstacles,
        "density": density,
        "obstacles_mask": obstacles_mask,
        "init_positions": init_positions,
        "sensing_radii": sensing_radii,
    }


def scenario_learning_comparison(
    output_dir: Path,
    seed: int,
    quick: bool,
) -> dict[str, Any]:
    """Reproduce Fig. 7/8 style comparison: best-response vs BLLL."""
    # Longer horizon for more meaningful late-stage comparison.
    runs = int(
        mode_value(
            quick=quick,
            quick_value=12,
            default_value=14,
        )
    )
    steps = int(
        mode_value(
            quick=quick,
            quick_value=180,
            default_value=1200,
        )
    )
    temperature = 0.2

    # Case A: homogeneous map without obstacles.
    width_a, height_a = 100, 100
    n_a = 15
    updates_per_step_a = n_a
    density_a = make_prior_density_map(width_a, height_a)
    init_a = left_edge_initial_positions(n_a, width_a, height_a, x=4, pad=3)
    radii_a = np.full(n_a, 10.0)

    # Case B: obstacle map.
    width_b, height_b = 100, 80
    n_b = 15
    updates_per_step_b = n_b
    obstacles_b = [
        RectObstacle(18, 14, 14, 20),
        RectObstacle(43, 8, 16, 24),
        RectObstacle(40, 44, 20, 14),
        RectObstacle(73, 31, 14, 28),
    ]
    mask_b = make_obstacle_mask(width_b, height_b, obstacles_b)
    density_b = make_uniform_density(width_b, height_b, obstacles_mask=mask_b)
    init_b = left_edge_initial_positions(n_b, width_b, height_b, x=4, pad=3, obstacles_mask=mask_b)
    radii_b = np.full(n_b, 9.0)

    def run_many_case_a(algorithm: str) -> tuple[np.ndarray, np.ndarray]:
        curves = []
        runtimes_s = []
        for r in range(runs):
            rng = np.random.default_rng(seed + 1000 + 31 * r)
            game = CoverageGame(density_a, radii_a, init_a)
            t0 = time.perf_counter()
            out = run_coverage_game(
                game=game,
                steps=steps,
                algorithm=algorithm,
                rng=rng,
                temperature=temperature,
                record_positions=False,
                updates_per_step=updates_per_step_a,
            )
            runtimes_s.append(time.perf_counter() - t0)
            curves.append(out.potentials)
        return np.asarray(curves), np.asarray(runtimes_s)

    def run_many_case_b(algorithm: str) -> tuple[np.ndarray, np.ndarray]:
        curves = []
        runtimes_s = []
        for r in range(runs):
            rng = np.random.default_rng(seed + 2000 + 37 * r)
            game = CoverageGame(
                density_b,
                radii_b,
                init_b,
                obstacles_mask=mask_b,
            )
            t0 = time.perf_counter()
            out = run_coverage_game(
                game=game,
                steps=steps,
                algorithm=algorithm,
                rng=rng,
                temperature=temperature,
                record_positions=False,
                obstacle_list=obstacles_b,
                updates_per_step=updates_per_step_b,
            )
            runtimes_s.append(time.perf_counter() - t0)
            curves.append(out.potentials)
        return np.asarray(curves), np.asarray(runtimes_s)

    blll_a, blll_a_runtime = run_many_case_a("blll")
    br_a, br_a_runtime = run_many_case_a("best_response")
    blll_b, blll_b_runtime = run_many_case_b("blll")
    br_b, br_b_runtime = run_many_case_b("best_response")

    # Runtime metric: mean wall-clock seconds per run (aggregated over both cases).
    blll_runtime_all = np.concatenate((blll_a_runtime, blll_b_runtime))
    br_runtime_all = np.concatenate((br_a_runtime, br_b_runtime))

    fig_path = output_dir / "fig7_8_learning_comparison.png"
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))

    x_time_a = time_axis(
        blll_a.shape[1],
        num_uavs=n_a,
        updates_per_step=updates_per_step_a,
    )
    x_time_b = time_axis(
        blll_b.shape[1],
        num_uavs=n_b,
        updates_per_step=updates_per_step_b,
    )

    for curve in br_a:
        axs[0, 0].plot(x_time_a, curve, color="tab:orange", alpha=0.22, lw=1.0)
    for curve in blll_a:
        axs[0, 0].plot(x_time_a, curve, color="tab:blue", alpha=0.22, lw=1.0)
    axs[0, 0].set_title("(a) Case A runs: BR (orange) vs BLLL (blue)")
    axs[0, 0].set_xlabel(time_xlabel(domain="learning"))
    axs[0, 0].set_ylabel("potential Φ(a)")
    axs[0, 0].grid(alpha=0.25)

    axs[0, 1].plot(x_time_a, np.mean(br_a, axis=0), color="tab:orange", lw=2.0, label="Best-response")
    axs[0, 1].plot(x_time_a, np.mean(blll_a, axis=0), color="tab:blue", lw=2.0, label="BLLL")
    axs[0, 1].set_title("(b) Case A mean curves")
    axs[0, 1].set_xlabel(time_xlabel(domain="learning"))
    axs[0, 1].set_ylabel("mean potential Φ(a)")
    axs[0, 1].grid(alpha=0.25)
    axs[0, 1].legend()

    for curve in br_b:
        axs[1, 0].plot(x_time_b, curve, color="tab:orange", alpha=0.22, lw=1.0)
    for curve in blll_b:
        axs[1, 0].plot(x_time_b, curve, color="tab:blue", alpha=0.22, lw=1.0)
    axs[1, 0].set_title("(c) Case B runs (obstacles): BR vs BLLL")
    axs[1, 0].set_xlabel(time_xlabel(domain="learning"))
    axs[1, 0].set_ylabel("potential Φ(a)")
    axs[1, 0].grid(alpha=0.25)

    axs[1, 1].plot(x_time_b, np.mean(br_b, axis=0), color="tab:orange", lw=2.0, label="Best-response")
    axs[1, 1].plot(x_time_b, np.mean(blll_b, axis=0), color="tab:blue", lw=2.0, label="BLLL")
    axs[1, 1].set_title("(d) Case B mean curves")
    axs[1, 1].set_xlabel(time_xlabel(domain="learning"))
    axs[1, 1].set_ylabel("mean potential Φ(a)")
    axs[1, 1].grid(alpha=0.25)
    axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return {
        "fig_path": fig_path,
        "case_a_final_blll": float(np.mean(blll_a[:, -1])),
        "case_a_final_br": float(np.mean(br_a[:, -1])),
        "case_b_final_blll": float(np.mean(blll_b[:, -1])),
        "case_b_final_br": float(np.mean(br_b[:, -1])),
        "runtime_blll_avg_s": float(np.mean(blll_runtime_all)),
        "runtime_br_avg_s": float(np.mean(br_runtime_all)),
        "runtime_blll_std_s": float(np.std(blll_runtime_all)),
        "runtime_br_std_s": float(np.std(br_runtime_all)),
    }


def scenario_cooperative_search(
    output_dir: Path,
    rng: np.random.Generator,
    quick: bool,
) -> dict[str, Any]:
    """Reproduce Fig. 9 and Fig. 10 style cooperative search trends."""
    # Slightly reduced map size for tractable runtime while preserving trends.
    width = int(
        mode_value(
            quick=quick,
            quick_value=50,
            default_value=70,
        )
    )
    height = int(
        mode_value(
            quick=quick,
            quick_value=50,
            default_value=70,
        )
    )

    # Paper-style base setting (Table 2 baseline).
    n = 10
    updates_per_step_main = n
    sensing_radius = 10.0
    comm_range = 40.0
    detection_prob = 0.9
    false_alarm_prob = 0.3
    sampling_frequency = 1
    temperature = 0.2
    gain_k = 1.0

    # Convergence-focused horizons by mode.
    main_steps = int(
        mode_value(
            quick=quick,
            quick_value=700,
            default_value=2500,
        )
    )
    # Fig. 9 snapshots: show early-to-mid transient only (0..500 paper timesteps)
    # even when simulation runs longer (e.g., 2500/5000 steps).
    num_snapshots = int(
        mode_value(
            quick=quick,
            quick_value=9,
            default_value=11,
        )
    )
    max_paper_t = min(
        500.0,
        time_value(
            main_steps,
            num_uavs=n,
            updates_per_step=updates_per_step_main,
        ),
    )
    snap_paper_t = np.linspace(0.0, max_paper_t, num=num_snapshots)
    snap_steps = np.rint(snap_paper_t * float(n) / float(updates_per_step_main)).astype(int)
    snap_steps = np.clip(snap_steps, 0, main_steps)
    snap_times = tuple(int(t) for t in np.unique(snap_steps))

    search_main = run_cooperative_search(
        width=width,
        height=height,
        num_uavs=n,
        sensing_radii=np.full(n, sensing_radius),
        comm_range=comm_range,
        detection_prob=detection_prob,
        false_alarm_prob=false_alarm_prob,
        sampling_frequency=sampling_frequency,
        steps=main_steps,
        temperature=temperature,
        uncertainty_gain=gain_k,
        rng=rng,
        snapshot_steps=snap_times,
        record_positions=True,
        record_every=1,
        updates_per_step=updates_per_step_main,
    )

    # Fig. 9 style probability snapshots for one vehicle.
    fig9_path = output_dir / "fig9_probability_snapshots.png"
    ordered_times = [t for t in snap_times if t in search_main.probability_snapshots]
    n_snaps = max(1, len(ordered_times))
    ncols = min(4, n_snaps)
    nrows = int(math.ceil(n_snaps / ncols))

    # Dynamic multi-row layout + dedicated colorbar column avoids overlap.
    fig_w = 4.1 * ncols + 1.0
    fig_h = 3.5 * nrows + 1.1
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    width_ratios = [1.0] * ncols + [0.07]
    gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=width_ratios, wspace=0.05, hspace=0.12)

    axs: list[plt.Axes] = []
    for r in range(nrows):
        for c in range(ncols):
            axs.append(fig.add_subplot(gs[r, c]))
    cax = fig.add_subplot(gs[:, -1])

    last_im = None
    for i_ax, ax in enumerate(axs):
        if i_ax < len(ordered_times):
            t = ordered_times[i_ax]
            p_map = search_main.probability_snapshots[t]
            last_im = ax.imshow(p_map, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
            t_disp = time_value(
                t,
                num_uavs=n,
                updates_per_step=updates_per_step_main,
            )
            ax.set_title(f"t = {t_disp:.1f}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("P(target present)")
    else:
        cax.axis("off")

    fig.suptitle("Fig. 9 reproduction: probability-map snapshots (vehicle 1)", fontsize=12)
    fig.savefig(fig9_path, dpi=180)
    plt.close(fig)

    traj_path = output_dir / "search_trajectories.png"
    # Visual background: prior density (for context)
    bg = make_prior_density_map(width, height)
    plot_trajectories(
        traj_path,
        bg,
        search_main.positions_history,
        title="Cooperative search: UAV trajectories",
    )

    gif_path = output_dir / "search_movement.gif"
    save_movement_gif(
        gif_path,
        bg,
        search_main.positions_history,
        title="Cooperative search movement",
        frame_stride=int(
            mode_value(
                quick=quick,
                quick_value=10,
                default_value=8,
            )
        ),
        fps=10,
    )

    # ------------------------------------------------------------------
    # Fig. 10 style parameter study
    # ------------------------------------------------------------------
    study_steps = int(
        mode_value(
            quick=quick,
            quick_value=300,
            default_value=1200,
        )
    )
    study_runs = int(
        mode_value(
            quick=quick,
            quick_value=6,
            default_value=8,
        )
    )

    case_specs: dict[str, list[Any]] = {
        "Case 1: different UAV counts n": [8, 10, 12],
        "Case 2: different sensing radius R_S": [8.0, 9.0, 10.0],
        "Case 3: different detection probability p_c": [0.7, 0.8, 0.9],
        "Case 4: different sampling frequency F_S": [2, 4, 6],
    }

    study_curves: dict[str, dict[str, np.ndarray]] = {k: {} for k in case_specs}

    for case_name, values in case_specs.items():
        for value in values:
            curves = []
            for r in range(study_runs):
                run_rng = np.random.default_rng(73000 + 101 * r + int(17 * (r + 1)))

                nn = n
                sr = np.full(n, sensing_radius)
                pc = detection_prob
                fs = sampling_frequency

                if case_name.startswith("Case 1"):
                    nn = int(value)
                    sr = np.full(nn, sensing_radius)
                elif case_name.startswith("Case 2"):
                    sr = np.full(n, float(value))
                elif case_name.startswith("Case 3"):
                    pc = float(value)
                elif case_name.startswith("Case 4"):
                    fs = int(value)

                updates_per_step_case = nn
                out = run_cooperative_search(
                    width=width,
                    height=height,
                    num_uavs=nn,
                    sensing_radii=sr,
                    comm_range=comm_range,
                    detection_prob=pc,
                    false_alarm_prob=false_alarm_prob,
                    sampling_frequency=fs,
                    steps=study_steps,
                    temperature=temperature,
                    uncertainty_gain=gain_k,
                    rng=run_rng,
                    snapshot_steps=(),
                    record_positions=False,
                    updates_per_step=updates_per_step_case,
                )
                curves.append(out.uncertainty_curve)

            mean_curve = np.mean(np.asarray(curves), axis=0)
            study_curves[case_name][str(value)] = mean_curve

    fig10_path = output_dir / "fig10_uncertainty_parameter_study.png"
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.2))
    axs_flat = axs.ravel()

    case_titles = list(case_specs.keys())
    for idx, case_name in enumerate(case_titles):
        ax = axs_flat[idx]
        for label, curve in study_curves[case_name].items():
            if case_name.startswith("Case 1"):
                nn_curve = int(float(label))
            else:
                nn_curve = n
            updates_curve = nn_curve
            x_time = time_axis(
                len(curve),
                num_uavs=nn_curve,
                updates_per_step=updates_curve,
            )
            ax.plot(x_time, curve, lw=1.8, label=label)
        ax.set_title(case_name)
        ax.set_xlabel(time_xlabel(domain="search"))
        ax.set_ylabel("average uncertainty")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Fig. 10 reproduction: uncertainty convergence under parameter changes")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(fig10_path, dpi=180)
    plt.close(fig)

    return {
        "fig9_path": fig9_path,
        "fig10_path": fig10_path,
        "traj_path": traj_path,
        "gif_path": gif_path,
        "final_uncertainty": float(search_main.uncertainty_curve[-1]),
        "uncertainty_curve": search_main.uncertainty_curve,
        "study_curves": study_curves,
    }


# -----------------------------------------------------------------------------
# Report generation (required by AGENTS.md)
# -----------------------------------------------------------------------------


def relative_to_report(path: Path, report_dir: Path) -> str:
    """Return a report-friendly relative path (supports sibling directories)."""
    return Path(os.path.relpath(path, start=report_dir)).as_posix()


def write_html_report(
    report_path: Path,
    artifacts: dict[str, dict[str, Any]],
    metrics: dict[str, float],
) -> None:
    """Write HTML report summarizing changes, visuals, and impact analysis."""
    report_dir = report_path.parent

    def rel(p: Path) -> str:
        return relative_to_report(p, report_dir)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hom = artifacts["homogeneous"]
    het = artifacts["heterogeneous"]
    obs = artifacts["obstacles"]
    cmp = artifacts["comparison"]
    sea = artifacts["search"]

    case_a_msg = (
        "BLLL reached a higher final mean potential in this case."
        if metrics["case_a_blll"] > metrics["case_a_br"]
        else "Best-response reached a higher final mean potential in this reconstruction."
    )
    case_b_msg = (
        "BLLL reached a higher final mean potential in this case."
        if metrics["case_b_blll"] > metrics["case_b_br"]
        else "Best-response reached a higher final mean potential in this reconstruction."
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Potential-Game UAV Simulation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px auto; max-width: 1500px; line-height: 1.5; }}
    h1, h2, h3 {{ margin-top: 1.1em; }}
    .viz-grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; }}
    figure.prominent {{ margin: 0; border: 2px solid #cfd6de; padding: 14px; border-radius: 10px; background: #fbfcfe; box-shadow: 0 3px 12px rgba(0,0,0,0.06); }}
    figure.prominent figcaption {{ font-size: 1.05rem; margin-bottom: 10px; font-weight: 600; }}
    img {{ width: 100%; height: auto; border-radius: 6px; border: 1px solid #e3e7ec; }}
    code {{ background: #f1f1f1; padding: 2px 5px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f3f5f7; }}
    .small {{ color: #444; font-size: 0.92em; }}
  </style>
</head>
<body>
  <h1>Simulation Report: Potential-Game Multi-UAV Cooperative Search</h1>
  <p class="small">Generated: {now}</p>

  <h2>1) Key reproduced visualizations</h2>
  <p class="small">Each figure is shown at large scale below (click to open full-size image).</p>
  <div class="viz-grid">
    <figure class="prominent">
      <figcaption>Homogeneous coverage + failure reconfiguration</figcaption>
      <a href="{rel(hom["fig_path"])}"><img src="{rel(hom["fig_path"])}" alt="homogeneous" /></a>
    </figure>
    <figure class="prominent">
      <figcaption>Heterogeneous coverage</figcaption>
      <a href="{rel(het["fig_path"])}"><img src="{rel(het["fig_path"])}" alt="heterogeneous" /></a>
    </figure>
    <figure class="prominent">
      <figcaption>Coverage with obstacles</figcaption>
      <a href="{rel(obs["fig_path"])}"><img src="{rel(obs["fig_path"])}" alt="obstacles" /></a>
    </figure>
    <figure class="prominent">
      <figcaption>BLLL vs best-response comparison</figcaption>
      <a href="{rel(cmp["fig_path"])}"><img src="{rel(cmp["fig_path"])}" alt="comparison" /></a>
    </figure>
    <figure class="prominent">
      <figcaption>Probability-map snapshots (search)</figcaption>
      <a href="{rel(sea["fig9_path"])}"><img src="{rel(sea["fig9_path"])}" alt="fig9" /></a>
    </figure>
    <figure class="prominent">
      <figcaption>Uncertainty convergence parameter study</figcaption>
      <a href="{rel(sea["fig10_path"])}"><img src="{rel(sea["fig10_path"])}" alt="fig10" /></a>
    </figure>
  </div>

  <h2>2) Agent movement over time</h2>
  <ul>
    <li>Homogeneous trajectories: <a href="{rel(hom["traj_path"])}">{rel(hom["traj_path"])}</a></li>
    <li>Homogeneous GIF: <a href="{rel(hom["gif_path"])}">{rel(hom["gif_path"])}</a></li>
    <li>Obstacle trajectories: <a href="{rel(obs["traj_path"])}">{rel(obs["traj_path"])}</a></li>
    <li>Obstacle GIF: <a href="{rel(obs["gif_path"])}">{rel(obs["gif_path"])}</a></li>
    <li>Search trajectories: <a href="{rel(sea["traj_path"])}">{rel(sea["traj_path"])}</a></li>
    <li>Search GIF: <a href="{rel(sea["gif_path"])}">{rel(sea["gif_path"])}</a></li>
  </ul>

  <h2>3) Impact analysis</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>Observed value</th><th>Interpretation</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Final homogeneous potential</td>
        <td>{metrics["homogeneous_final"]:.4f}</td>
        <td>BLLL converges to high-coverage states and recovers after failures.</td>
      </tr>
      <tr>
        <td>Final heterogeneous potential</td>
        <td>{metrics["heterogeneous_final"]:.4f}</td>
        <td>Method naturally handles mixed sensing radii.</td>
      </tr>
      <tr>
        <td>Final obstacle potential</td>
        <td>{metrics["obstacle_final"]:.4f}</td>
        <td>Constrained action sets support non-convex environments.</td>
      </tr>
      <tr>
        <td>Case A mean final potential (BLLL / BR)</td>
        <td>{metrics["case_a_blll"]:.4f} / {metrics["case_a_br"]:.4f}</td>
        <td>{case_a_msg}</td>
      </tr>
      <tr>
        <td>Case B mean final potential (BLLL / BR)</td>
        <td>{metrics["case_b_blll"]:.4f} / {metrics["case_b_br"]:.4f}</td>
        <td>{case_b_msg}</td>
      </tr>
      <tr>
        <td>Average runtime per run (BLLL / BR)</td>
        <td>{metrics["runtime_blll_avg_s"]:.4f}s / {metrics["runtime_br_avg_s"]:.4f}s</td>
        <td>Wall-clock mean over both comparison cases (A and B).</td>
      </tr>
      <tr>
        <td>Final average uncertainty (search)</td>
        <td>{metrics["search_uncertainty_final"]:.6f}</td>
        <td>Uncertainty decreases toward zero as observations + fusion accumulate.</td>
      </tr>
    </tbody>
  </table>

  <p class="small">
    Reproduction note: numerical values may differ from the paper due to unavailable exact map equations,
    random seeds, and implementation-level assumptions reconstructed from parsed equations.
    Qualitative behaviors and method structure are preserved.
  </p>
</body>
</html>
"""

    report_path.write_text(html)


def write_html_presentation_report(
    report_path: Path,
    artifacts: dict[str, dict[str, Any]],
    metrics: dict[str, float],
) -> None:
    """Write a presentation-oriented HTML report (large slide-like visuals)."""
    report_dir = report_path.parent

    def rel(p: Path) -> str:
        return relative_to_report(p, report_dir)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hom = artifacts["homogeneous"]
    het = artifacts["heterogeneous"]
    obs = artifacts["obstacles"]
    cmp = artifacts["comparison"]
    sea = artifacts["search"]

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Presentation: Potential-Game Multi-UAV Results</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color: #111827; background: #f8fafc; }}
    .slide {{ min-height: 100vh; box-sizing: border-box; padding: 28px 5vw; border-bottom: 1px solid #dbe3ee; background: #ffffff; }}
    .slide h1 {{ margin: 0 0 10px 0; font-size: 2.1rem; }}
    .slide h2 {{ margin: 0 0 12px 0; font-size: 1.7rem; }}
    .subtitle {{ color: #475569; font-size: 1.05rem; margin-bottom: 14px; }}
    .small {{ color: #64748b; font-size: 0.95rem; }}
    .visual {{ width: 100%; max-height: 78vh; object-fit: contain; border: 1px solid #cbd5e1; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); background: #fff; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 18px; }}
    .kpi {{ border: 1px solid #d7e0eb; border-radius: 10px; padding: 14px; background: #f8fbff; }}
    .kpi .label {{ font-size: 0.9rem; color: #475569; }}
    .kpi .value {{ margin-top: 6px; font-size: 1.45rem; font-weight: 700; }}
    ul {{ line-height: 1.6; }}
    a {{ color: #0f4fa8; }}
  </style>
</head>
<body>
  <section class="slide">
    <h1>Potential-Game Multi-UAV Cooperative Search</h1>
    <div class="subtitle">Presentation mode report (large visuals)</div>
    <div class="small">Generated: {now}</div>
  </section>

  <section class="slide">
    <h2>Homogeneous coverage + failure reconfiguration</h2>
    <img class="visual" src="{rel(hom["fig_path"])}" alt="homogeneous" />
  </section>

  <section class="slide">
    <h2>Heterogeneous coverage</h2>
    <img class="visual" src="{rel(het["fig_path"])}" alt="heterogeneous" />
  </section>

  <section class="slide">
    <h2>Coverage with obstacles</h2>
    <img class="visual" src="{rel(obs["fig_path"])}" alt="obstacles" />
  </section>

  <section class="slide">
    <h2>BLLL vs best-response comparison</h2>
    <img class="visual" src="{rel(cmp["fig_path"])}" alt="comparison" />
  </section>

  <section class="slide">
    <h2>Probability-map snapshots (search)</h2>
    <img class="visual" src="{rel(sea["fig9_path"])}" alt="fig9" />
  </section>

  <section class="slide">
    <h2>Uncertainty convergence parameter study</h2>
    <img class="visual" src="{rel(sea["fig10_path"])}" alt="fig10" />
  </section>

  <section class="slide">
    <h2>Agent movement assets</h2>
    <ul>
      <li>Homogeneous trajectories: <a href="{rel(hom["traj_path"])}">{rel(hom["traj_path"])}</a></li>
      <li>Homogeneous GIF: <a href="{rel(hom["gif_path"])}">{rel(hom["gif_path"])}</a></li>
      <li>Obstacle trajectories: <a href="{rel(obs["traj_path"])}">{rel(obs["traj_path"])}</a></li>
      <li>Obstacle GIF: <a href="{rel(obs["gif_path"])}">{rel(obs["gif_path"])}</a></li>
      <li>Search trajectories: <a href="{rel(sea["traj_path"])}">{rel(sea["traj_path"])}</a></li>
      <li>Search GIF: <a href="{rel(sea["gif_path"])}">{rel(sea["gif_path"])}</a></li>
    </ul>
  </section>

  <section class="slide">
    <h2>Impact metrics</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="label">Final homogeneous potential</div><div class="value">{metrics["homogeneous_final"]:.4f}</div></div>
      <div class="kpi"><div class="label">Final heterogeneous potential</div><div class="value">{metrics["heterogeneous_final"]:.4f}</div></div>
      <div class="kpi"><div class="label">Final obstacle potential</div><div class="value">{metrics["obstacle_final"]:.4f}</div></div>
      <div class="kpi"><div class="label">Case A mean final (BLLL / BR)</div><div class="value">{metrics["case_a_blll"]:.4f} / {metrics["case_a_br"]:.4f}</div></div>
      <div class="kpi"><div class="label">Case B mean final (BLLL / BR)</div><div class="value">{metrics["case_b_blll"]:.4f} / {metrics["case_b_br"]:.4f}</div></div>
      <div class="kpi"><div class="label">Avg runtime per run (BLLL / BR)</div><div class="value">{metrics["runtime_blll_avg_s"]:.4f}s / {metrics["runtime_br_avg_s"]:.4f}s</div></div>
      <div class="kpi"><div class="label">Final search uncertainty</div><div class="value">{metrics["search_uncertainty_final"]:.6f}</div></div>
    </div>
    <p class="small">Reproduction note: values may differ from the paper due to unavailable exact map equations and stochastic effects.</p>
  </section>
</body>
</html>
"""

    report_path.write_text(html)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce potential-game UAV cooperative search simulations using paper timestep and paper semantics."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster, lower-cost version (fewer steps/runs).",
    )

    parser.add_argument(
        "--presentation-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default. Generate an additional presentation-style HTML report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated figures and animations.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated HTML report(s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.report_dir)

    rng = np.random.default_rng(args.seed)

    run_mode = "quick" if args.quick else "default"
    timeline_mode = "paper-timestep"
    dynamics_mode = "paper-step-semantics"
    print(f"Running mode: {run_mode} | timeline: {timeline_mode} | dynamics: {dynamics_mode}")

    print("[1/6] Running homogeneous coverage scenario...")
    hom = scenario_homogeneous_with_failures(
        args.output_dir,
        rng,
        quick=args.quick,
    )

    print("[2/6] Running heterogeneous coverage scenario...")
    het = scenario_heterogeneous_coverage(
        args.output_dir,
        rng,
        quick=args.quick,
    )

    print("[3/6] Running obstacle coverage scenario...")
    obs = scenario_obstacle_coverage(
        args.output_dir,
        rng,
        quick=args.quick,
    )

    print("[4/6] Running BLLL vs best-response comparisons...")
    cmp = scenario_learning_comparison(
        args.output_dir,
        seed=args.seed,
        quick=args.quick,
    )

    print("[5/6] Running cooperative search scenarios...")
    sea = scenario_cooperative_search(
        args.output_dir,
        rng,
        quick=args.quick,
    )

    artifacts = {
        "homogeneous": hom,
        "heterogeneous": het,
        "obstacles": obs,
        "comparison": cmp,
        "search": sea,
    }

    metrics = {
        "homogeneous_final": hom["final_potential"],
        "heterogeneous_final": het["final_potential"],
        "obstacle_final": obs["final_potential"],
        "case_a_blll": cmp["case_a_final_blll"],
        "case_a_br": cmp["case_a_final_br"],
        "case_b_blll": cmp["case_b_final_blll"],
        "case_b_br": cmp["case_b_final_br"],
        "runtime_blll_avg_s": cmp["runtime_blll_avg_s"],
        "runtime_br_avg_s": cmp["runtime_br_avg_s"],
        "search_uncertainty_final": sea["final_uncertainty"],
    }

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "quick" if args.quick else "default"
    report_path = args.report_dir / f"simulation_report_{mode_tag}_{timestamp}.html"

    print("[6/6] Writing HTML report...")
    write_html_report(
        report_path=report_path,
        artifacts=artifacts,
        metrics=metrics,
    )

    presentation_report_path: Path | None = None
    if args.presentation_mode:
        presentation_report_path = args.report_dir / (f"simulation_presentation_{mode_tag}_{timestamp}.html")
        print("[6b] Writing presentation-mode HTML report...")
        write_html_presentation_report(
            report_path=presentation_report_path,
            artifacts=artifacts,
            metrics=metrics,
        )

    print("Done.")
    print(f"- Figures/animations: {args.output_dir.resolve()}")
    print(f"- Report: {report_path.resolve()}")
    if presentation_report_path is not None:
        print(f"- Presentation report: {presentation_report_path.resolve()}")
    print("- Parsed paper extraction artifacts: paper_extraction/")


if __name__ == "__main__":
    main()
