import os, sys, glob
import numpy as np

TARGET      = np.array([0.0, 60.0, 0.0])
DIST_THRESH = 10    # meters
SPEED_THRESH= 2   # m/s
Y_FAIL      = 59.0   # if y < Y_FAIL, automatic failure

def summarize(arr):
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "std":    float(np.std(arr)),
    }

def analyze(model_name, root_dir="."):
    runs_dir = os.path.join(os.getcwd(),root_dir, "runs",       model_name)
    vel_dir  = os.path.join(os.getcwd(), root_dir, "velocities", model_name)
    run_fps  = sorted(glob.glob(os.path.join(runs_dir, "chaser*.txt")))
    vel_fps  = sorted(glob.glob(os.path.join(vel_dir,  "chaser*.txt")))

    if not run_fps or not vel_fps:
        raise FileNotFoundError("Missing runs/ or velocities/ for: " + model_name)

    N = min(len(run_fps), len(vel_fps))
    dock_pos   = np.zeros((N, 3))
    dock_vel   = np.zeros((N, 3))
    dock_speed = np.zeros(N)
    dock_time  = np.zeros(N, dtype=int)
    success    = np.zeros(N, dtype=bool)

    for i, (r_fp, v_fp) in enumerate(zip(run_fps, vel_fps)):
        pos_traj = np.loadtxt(r_fp, delimiter=",").reshape(-1, 3)
        vel_traj = np.loadtxt(v_fp, delimiter=",").reshape(-1, 3)
        T = pos_traj.shape[0]

        # scan for docking event
        dock_step = None
        for t in range(T):
            p = pos_traj[t]
            v = vel_traj[t]
            if p[1] < Y_FAIL:
                # below y-threshold: immediate failure, stop scanning
                dock_step = T-1
                success[i] = False
                break
            dist  = np.linalg.norm(p - TARGET)
            speed = np.linalg.norm(v)
            if (dist < DIST_THRESH) and (speed < SPEED_THRESH):
                dock_step = t
                success[i] = True
                break

        if dock_step is None:
            # never breached Y_FAIL but never docked either
            dock_step = T-1
            success[i] = False

        dock_pos[i]   = pos_traj [dock_step]
        dock_vel[i]   = vel_traj [dock_step]
        dock_speed[i] = np.linalg.norm(dock_vel[i])
        dock_time [i] = dock_step

        status = "SUCCESS" if success[i] else "FAIL"
        print(f"{os.path.basename(r_fp):<15s}  {status:7s}  "
              f"step={dock_step:4d}  y={dock_pos[i,1]:6.2f}  "
              f"dist={np.linalg.norm(dock_pos[i]-TARGET):6.3f}  "
              f"speed={dock_speed[i]:6.4f}")

    # compile summary
    summary = {
        "x":      summarize(dock_pos[:, 0]),
        "y":      summarize(dock_pos[:, 1]),
        "z":      summarize(dock_pos[:, 2]),
        "xdot":   summarize(dock_vel[:, 0]),
        "ydot":   summarize(dock_vel[:, 1]),
        "zdot":   summarize(dock_vel[:, 2]),
        "speed":  summarize(dock_speed),
        "time":   summarize(dock_time),
        "success_rate": float(success.sum()) / N,
        "counts": {
            "total":     N,
            "successes": int(success.sum()),
            "failures":  int(N - success.sum())
        }
    }
    return summary

def print_summary(summary, model_name):
    print(f"\n=== Summary for '{model_name}' ===\n")
    for key in ("x", "y", "z", "xdot", "ydot", "zdot", "speed", "time"):
        print(f"[{key}]")
        for stat, val in summary[key].items():
            print(f"  {stat:7s} = {val}")
        print()
    cr = summary["success_rate"]
    cnt= summary["counts"]
    print(f"Success rate: {cr*100:.2f}% ({cnt['successes']}/{cnt['total']})\n")

if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 3):
        print(f"Usage: {sys.argv[0]} <model_name> [<root_dir>]")
        sys.exit(1)

    model_name = sys.argv[1]
    root_dir   = sys.argv[2] if len(sys.argv)==3 else "."

    stats = analyze(model_name, root_dir)
    print_summary(stats, model_name)