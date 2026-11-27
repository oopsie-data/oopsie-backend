"""Visualize converted oopsiedata_format_v1 episodes.
NOTE(zhouzypaul): I've made this and only tested on the SOAR dataset, and it only deals with EEF action space.

Two modes:

  Episode view — plots trajectory data + video frames for one episode:
    python -m oopsie_backend.tools.visualize_episodes --episode path/to/episode.h5
    python -m oopsie_backend.tools.visualize_episodes --episode path/to/episode.h5 --out plot.png

  Stats view — dataset-level summary across a whole directory:
    python -m oopsie_backend.tools.visualize_episodes --stats path/to/soar_converted/
    python -m oopsie_backend.tools.visualize_episodes --stats path/to/soar_converted/ --out stats.png

  In both modes, --out saves to file instead of opening an interactive window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np

from oopsie_backend.common.h5 import is_h5

matplotlib.use("Agg")  # non-interactive by default; overridden below for display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── helpers ──────────────────────────────────────────────────────────────────

def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return _decode(v.flat[0]) if v.size else ""
    return str(v)


def _load_episode(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        attrs = {k: _decode(f.attrs[k]) for k in f.attrs}
        try:
            profile = json.loads(attrs.get("robot_profile", "{}"))
        except json.JSONDecodeError:
            profile = {}

        robot_states = {k: f[f"observations/robot_states/{k}"][()] for k in f["observations/robot_states"]}
        actions = {
            k: f[f"actions/{k}"][()]
            for k in f["actions"]
            if f[f"actions/{k}"].shape is not None  # skip h5py.Empty
        }
        video_paths = {k: _decode(f[f"observations/video_paths/{k}"][()]) for k in f["observations/video_paths"]}
        annotations = {}
        if "episode_annotations" in f:
            for ann_name in f["episode_annotations"]:
                annotations[ann_name] = {k: f[f"episode_annotations/{ann_name}"].attrs[k] for k in f[f"episode_annotations/{ann_name}"].attrs}

    return dict(attrs=attrs, profile=profile, robot_states=robot_states,
                actions=actions, video_paths=video_paths, annotations=annotations)


def _extract_frames(mp4_path: Path, n: int = 5) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ── episode view ──────────────────────────────────────────────────────────────

def plot_episode(h5_path: Path, out: Path | None) -> None:
    ep = _load_episode(h5_path)
    attrs = ep["attrs"]
    profile = ep["profile"]
    rs = ep["robot_states"]
    acts = ep["actions"]
    anns = ep["annotations"]
    vid_paths = ep["video_paths"]

    T = next(iter(rs.values())).shape[0]
    t = np.arange(T)
    control_freq = float(profile.get("control_freq", 15))
    t_sec = t / control_freq

    success_str = ""
    for ann_name, ann_attrs in anns.items():
        if "success" in ann_attrs:
            s = float(ann_attrs["success"])
            success_str = f"  [{ann_name}: {'SUCCESS' if s >= 0.5 else 'FAILURE'} ({s:.2f})]"

    lang = attrs.get("language_instruction", "")
    policy = profile.get("policy_name", "")
    title = f"{lang}{success_str}\n{h5_path.name}  |  policy={policy}  T={T}  ({T/control_freq:.1f}s)"

    # Video frames for the top strip
    video_frames: list[np.ndarray] = []
    for cam, rel_path in vid_paths.items():
        abs_path = h5_path.parent / rel_path
        if abs_path.is_file():
            frames = _extract_frames(abs_path, n=5)
            video_frames = frames  # use first camera
            break

    n_vid = len(video_frames)
    # Layout: video strip (if present) + 3 trajectory panels (cart xyz, rpy, gripper)
    n_traj_rows = 3
    total_rows = (1 if n_vid else 0) + n_traj_rows

    fig = plt.figure(figsize=(16, 4 * total_rows))
    fig.suptitle(title, fontsize=10, wrap=True)
    gs = gridspec.GridSpec(total_rows, max(n_vid, 1), figure=fig, hspace=0.45, wspace=0.08)

    row = 0
    # ── video strip ──
    if n_vid:
        frame_indices = np.linspace(0, T - 1, n_vid, dtype=int)
        for i, (frame, fi) in enumerate(zip(video_frames, frame_indices)):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(frame)
            ax.set_title(f"t={fi} ({fi/control_freq:.1f}s)", fontsize=8)
            ax.axis("off")
        row = 1

    # ── cartesian position xyz ──
    ax_xyz = fig.add_subplot(gs[row, :])
    cart_obs = rs.get("cartesian_position")
    cart_act = acts.get("cartesian_position")
    labels_xyz = ["x", "y", "z"]
    colors_xyz = ["tab:red", "tab:green", "tab:blue"]
    for i, (lbl, col) in enumerate(zip(labels_xyz, colors_xyz)):
        if cart_obs is not None and cart_obs.shape[1] > i:
            ax_xyz.plot(t_sec, cart_obs[:, i], color=col, alpha=0.8, lw=1.2, label=f"obs {lbl}")
        if cart_act is not None and cart_act.shape[1] > i:
            ax_xyz.plot(t_sec, cart_act[:, i], color=col, alpha=0.4, lw=1, ls="--", label=f"act {lbl}")
    ax_xyz.set_ylabel("position (m)")
    ax_xyz.set_title("Cartesian XYZ  (solid=obs, dashed=action/next-state)")
    ax_xyz.legend(ncol=6, fontsize=7, loc="upper right")
    ax_xyz.grid(True, alpha=0.3)
    row += 1

    # ── cartesian orientation rpy ──
    ax_rpy = fig.add_subplot(gs[row, :])
    labels_rpy = ["roll", "pitch", "yaw"]
    colors_rpy = ["tab:orange", "tab:purple", "tab:brown"]
    for i, (lbl, col) in enumerate(zip(labels_rpy, colors_rpy)):
        idx = i + 3
        if cart_obs is not None and cart_obs.shape[1] > idx:
            ax_rpy.plot(t_sec, cart_obs[:, idx], color=col, alpha=0.8, lw=1.2, label=f"obs {lbl}")
        if cart_act is not None and cart_act.shape[1] > idx:
            ax_rpy.plot(t_sec, cart_act[:, idx], color=col, alpha=0.4, lw=1, ls="--", label=f"act {lbl}")
    ax_rpy.set_ylabel("angle (rad)")
    ax_rpy.set_title("Cartesian Orientation RPY  (solid=obs, dashed=action/next-state)")
    ax_rpy.legend(ncol=6, fontsize=7, loc="upper right")
    ax_rpy.grid(True, alpha=0.3)
    row += 1

    # ── gripper ──
    ax_grip = fig.add_subplot(gs[row, :])
    grip_obs = rs.get("gripper_position")
    grip_act = acts.get("gripper_position")
    if grip_obs is not None:
        ax_grip.plot(t_sec, grip_obs[:, 0], color="tab:cyan", lw=1.5, label="obs gripper")
    if grip_act is not None:
        ax_grip.plot(t_sec, grip_act[:, 0], color="tab:cyan", alpha=0.4, lw=1, ls="--", label="act gripper (next-state)")
    ax_grip.set_ylabel("gripper state")
    ax_grip.set_xlabel("time (s)")
    ax_grip.set_title("Gripper  (solid=obs, dashed=action/next-state)")
    ax_grip.legend(fontsize=7, loc="upper right")
    ax_grip.grid(True, alpha=0.3)

    _save_or_show(fig, out)


# ── stats view ────────────────────────────────────────────────────────────────

def plot_stats(data_dir: Path, out: Path | None) -> None:
    h5_files = sorted(p for p in data_dir.glob("*") if is_h5(p))
    if not h5_files:
        print(f"No HDF5 files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    n_total = len(h5_files)
    print(f"Scanning {n_total} episodes...", end=" ", flush=True)

    traj_lengths: list[int] = []
    successes: list[float] = []
    robots: dict[str, int] = {}
    policies: dict[str, int] = {}
    tasks: dict[str, int] = {}

    for p in h5_files:
        try:
            with h5py.File(p, "r") as f:
                # trajectory length from first robot_state array
                rs = f["observations/robot_states"]
                T = next(iter(rs.values())).shape[0]
                traj_lengths.append(T)

                # success
                for ann_name in f.get("episode_annotations", {}).keys():
                    attrs = f[f"episode_annotations/{ann_name}"].attrs
                    if "success" in attrs:
                        successes.append(float(attrs["success"]))
                        break

                # robot / policy / task from episode_id
                ep_id = _decode(f.attrs.get("episode_id", ""))
                parts = ep_id.split("_")  # soar_NNNNNN_berkeley_robot_X_task..._policy..._date..._outcome_trajN
                # structure: soar, counter, berkeley, robot, X, <task parts>, <policy>, <date>, <outcome>, trajN
                if len(parts) > 4 and parts[0] == "soar":
                    robot_id = parts[4] if len(parts) > 4 else "?"
                    robots[robot_id] = robots.get(robot_id, 0) + 1

                profile_str = _decode(f.attrs.get("robot_profile", "{}"))
                try:
                    profile = json.loads(profile_str)
                    policy = profile.get("policy_name", "unknown")
                    policies[policy] = policies.get(policy, 0) + 1
                except json.JSONDecodeError:
                    pass

                lang = _decode(f.attrs.get("language_instruction", ""))
                if lang:
                    tasks[lang] = tasks.get(lang, 0) + 1

        except Exception:
            pass

    print("done.")

    control_freq = 5.0  # NOTE(zhouzypaul): hard coded for now
    durations = [t / control_freq for t in traj_lengths]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"SOAR → Oopsie dataset  |  {n_total:,} episodes  |  {data_dir}", fontsize=11)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Trajectory length histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(traj_lengths, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Trajectory length (steps)")
    ax1.set_ylabel("Episodes")
    ax1.set_title("Trajectory length distribution")
    ax1.axvline(np.median(traj_lengths), color="orange", lw=1.5, label=f"median={int(np.median(traj_lengths))}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Duration histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(durations, bins=50, color="mediumseagreen", edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Duration (seconds)")
    ax2.set_ylabel("Episodes")
    ax2.set_title(f"Episode duration (at {control_freq:.0f} Hz)")
    ax2.axvline(np.median(durations), color="orange", lw=1.5, label=f"median={np.median(durations):.1f}s")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Success / failure pie
    ax3 = fig.add_subplot(gs[0, 2])
    if successes:
        n_succ = sum(1 for s in successes if s >= 0.5)
        n_fail = len(successes) - n_succ
        wedges, texts, autotexts = ax3.pie(
            [n_succ, n_fail],
            labels=[f"Success\n({n_succ:,})", f"Failure\n({n_fail:,})"],
            colors=["#4CAF50", "#F44336"],
            autopct="%1.1f%%",
            startangle=90,
        )
        for at in autotexts:
            at.set_fontsize(9)
    ax3.set_title("Success / Failure")

    # 4. Per-robot episode count
    ax4 = fig.add_subplot(gs[1, 0])
    if robots:
        sorted_robots = sorted(robots.items())
        rx = [f"robot_{k}" for k, _ in sorted_robots]
        ry = [v for _, v in sorted_robots]
        bars = ax4.bar(rx, ry, color="steelblue")
        for bar, v in zip(bars, ry):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{v:,}", ha="center", va="bottom", fontsize=7)
        ax4.set_ylabel("Episodes")
        ax4.set_title("Episodes per robot")
        ax4.grid(True, alpha=0.3, axis="y")

    # 5. Policy distribution (top 12)
    ax5 = fig.add_subplot(gs[1, 1])
    if policies:
        top = sorted(policies.items(), key=lambda x: -x[1])[:12]
        px = [p for p, _ in top]
        py = [v for _, v in top]
        bars = ax5.barh(px[::-1], py[::-1], color="mediumpurple")
        for bar, v in zip(bars, py[::-1]):
            ax5.text(bar.get_width() + max(py) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{v:,}", va="center", fontsize=7)
        ax5.set_xlabel("Episodes")
        ax5.set_title("Policy distribution (top 12)")
        ax5.grid(True, alpha=0.3, axis="x")

    # 6. Top tasks
    ax6 = fig.add_subplot(gs[1, 2])
    if tasks:
        top_tasks = sorted(tasks.items(), key=lambda x: -x[1])[:12]
        tx = [t[:40] + ("…" if len(t) > 40 else "") for t, _ in top_tasks]
        ty = [v for _, v in top_tasks]
        bars = ax6.barh(tx[::-1], ty[::-1], color="darkorange")
        for bar, v in zip(bars, ty[::-1]):
            ax6.text(bar.get_width() + max(ty) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{v:,}", va="center", fontsize=7)
        ax6.set_xlabel("Episodes")
        ax6.set_title("Top tasks (by language instruction)")
        ax6.tick_params(axis="y", labelsize=7)
        ax6.grid(True, alpha=0.3, axis="x")

    _save_or_show(fig, out)


# ── output helper ─────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, out: Path | None) -> None:
    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SOAR → Oopsie converted episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", "-e", type=Path, metavar="FILE",
                       help="Path to a single .h5 episode file.")
    group.add_argument("--stats", "-s", type=Path, metavar="DIR",
                       help="Path to a directory of .h5 files for dataset-level stats.")
    parser.add_argument("--out", "-o", type=Path, default=None,
                        help="Save plot to this path (PNG/PDF). Omit to open an interactive window.")
    args = parser.parse_args()

    if args.episode:
        if not args.episode.is_file():
            parser.error(f"File not found: {args.episode}")
        plot_episode(args.episode, args.out)
    else:
        if not args.stats.is_dir():
            parser.error(f"Directory not found: {args.stats}")
        plot_stats(args.stats, args.out)


if __name__ == "__main__":
    main()
