import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_all_chasers(
    type_folder = 'Benchmark',
    name_model='vbar-agentFirst_1',
    half_angle_deg=20,
    cone_height=200,
    interval=50,
    only_initial=False,
    save_animation = True,
    save_path = 'firstAnimation.gif',
    fps = 20
):
    """
    Animate or scatter chaser trajectories in `runs/vbar{name_folder}`.

    Parameters:
    -----------
    name_folder : int
        Numeric suffix for vbar folder (e.g. 31 loads `runs/vbar31`).
    half_angle_deg : float
        Half-angle (in degrees) of the fixed LoS cone.
    cone_height : float
        Height of the LoS cone along +Z axis.
    interval : int
        Delay between frames in milliseconds.
    only_initial : bool
        If True, render a static scatter of initial states only.
    """
    # 1) Build and find paths
    base_dir = os.path.join(os.getcwd(), f"{type_folder}/runs/{name_model}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Folder not found: {base_dir}")
    pattern = os.path.join(base_dir, "chaser*.txt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No chaser files found in {base_dir}")
    print(f"Found {len(paths)} trajectories in {base_dir}")

    # 2) Load trajectories
    trajectories = [np.loadtxt(p, delimiter=",") for p in paths]

    # 3) Set up 3D figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # 4) Draw static LoS cone
    theta = np.deg2rad(half_angle_deg)
    zs_cone = np.linspace(0, cone_height, 50)
    for z in zs_cone:
        r = z * np.tan(theta)
        phi = np.linspace(0, 2*np.pi, 60)
        xs = r * np.cos(phi); ys = r * np.sin(phi)
        ax.plot(xs, ys, z, color="gray", alpha=0.3)

    # 5) Initial-only static scatter
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    if only_initial:
        initials = np.array([traj[0] for traj in trajectories])
        ax.scatter(
            initials[:, 0], initials[:, 1], initials[:, 2],
            c=colors, s=50, depthshade=True
        )
        ax.set_title(f"Initial States for {name_model}")
        plt.tight_layout()
        plt.show()
        return

    # 6) Full animation of trajectories over time
    num_frames = max(traj.shape[0] for traj in trajectories)
    all_pts = np.vstack(trajectories)
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.set_title(f"All Chasers {name_model} + LoS Cone")

    # prepare dynamic line and marker artists
    lines, pts = [], []
    for c in colors:
        ln, = ax.plot([], [], [], lw=1.5, color=c)
        pt, = ax.plot([], [], [], 'o', ms=4, color=c)
        lines.append(ln)
        pts.append(pt)

    def init():
        for ln, pt in zip(lines, pts):
            ln.set_data([], [])
            ln.set_3d_properties([])
            pt.set_data([], [])
            pt.set_3d_properties([])
        return lines + pts

    def update(i):
        for traj, ln, pt in zip(trajectories, lines, pts):
            if i < traj.shape[0]:
                x, y, z = traj[:i+1].T
                ln.set_data(x, y); ln.set_3d_properties(z)
                pt.set_data(x[-1:], y[-1:]); pt.set_3d_properties(z[-1:])
        return lines + pts

    ani = FuncAnimation(
        fig, update, frames=num_frames,
        init_func=init, blit=True, interval=interval
    )
    plt.tight_layout()
    if save_animation:
        # choose the PillowWriter
        writer = PillowWriter(fps=fps)
        ani.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
    return ani



if __name__ == "__main__":
    # prompt for vbar folder number
    try:
        type_folder_input = input('Training or Benchmark?')
        name_model_input = input('Name of the model')
    except ValueError:
        print("Error: folder number must be an integer.")
        sys.exit(1)

    # prompt for initial-only option
    only_flag = input("Render only initial states? [Y/N]: ").strip().lower()
    only_initial = True if only_flag == 'y' else False

    animate_all_chasers(
        type_folder=type_folder_input,
        name_model=name_model_input,
        only_initial=only_initial
    )
