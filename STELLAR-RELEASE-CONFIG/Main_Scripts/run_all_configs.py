import os, glob, yaml, argparse
from typing import Any, Dict

# your train/bench/analyze/animate imports
from finetune_model import train_PPO, CHECKBACK
from run_benchmark  import benchmark_PPO
from compute_model_stats import analyze
from animate_chaser import animate_all_chasers

# import any classes you need to map from strings
from finetune_model import  TimeLimitWrapper  
from torch.nn import GELU  

CLASS_MAP = {
  "TimeLimitWrapper": TimeLimitWrapper,
  "GELU":            GELU,
}

def _set_by_path(d: dict, path: str, val: Any):
    """
    Given d={"a": {"b": {...}}}, path="a.b.c", set d["a"]["b"]["c"]=val.
    Creates intermediate dicts if needed.
    """
    keys = path.split(".")
    sub = d
    for k in keys[:-1]:
        sub = sub.setdefault(k, {})
    sub[keys[-1]] = val


def find_latest_run_folder(root_dir: str, model_name: str) -> str:
    """
    Given e.g. root_dir="Benchmark" and model_name="vbar-test",
    finds the subfolder under Benchmark/runs/ whose name is
    'vbar-test_<id>' with the largest <id>.
    Returns that folder name (not the full path).
    """
    pattern = os.path.join(os.getcwd(),root_dir, "runs", f"{model_name}_*")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No runs found matching {pattern}")
    # Extract the numeric suffix and pick the max
    def run_id(path):
        base = os.path.basename(path)
        suffix = base.split("_")[-1]
        return int(suffix) if suffix.isdigit() else -1

    latest = max(candidates, key=run_id)
    return os.path.basename(latest)



def train_benchmark_analyze(
    model_name: str,
    timesteps: int,
    episodes: int,
    stats_path: str,
    train_max_steps: int,
    load_model: str,
    bench_max_steps: int,
    train_env_kwargs: dict,
    train_gym_kwargs: dict,
    bench_env_kwargs: dict,
    bench_gym_kwargs: dict,
    ppo_kwargs: dict,
    policy_dict: dict,
    plot_trajectories: bool,
    plot_half_angle_deg: float,
    plot_cone_height: float,
    callback=CHECKBACK,
):
    """
    1) Train   (with `callback`)
    2) Benchmark
    3) Analyze
    """

    # 1) TRAIN
    print(f"\n→ [1/3] Training '{model_name}' for {timesteps} steps …")
    train_PPO(
        model_name       = f'{model_name}',
        stats_file       = os.path.join(os.getcwd(), stats_path) if stats_path is not False else False,
        total_timesteps  = timesteps,
        load_model       = load_model,
        ppo_kwargs       = ppo_kwargs,
        env_kwargs       = train_env_kwargs,
        gym_kwargs       = train_gym_kwargs,
        wrapper_kwargs   = {"max_steps": train_max_steps},
        callback         = callback,
        policy_dict      = policy_dict
    )

    # load the new normalizer

    models_dir = os.path.join(os.getcwd(), 'models')
    # pattern = os.path.join(models_dir, f"{model_name}_*.zip")
    # existing = glob.glob(pattern)
    # idxs = []
    # for path in existing:
    #     base = os.path.basename(path)
    #     # split off the trailing "_<id>.zip"
    #     parts = base.rsplit("_", 1)
    #     if len(parts) == 2 and parts[1].endswith(".zip"):
    #         num = parts[1][:-4]   # remove ".zip"
    #         if num.isdigit():
    #             idxs.append(int(num))
    # run_id = max(idxs) + 1 if idxs else 1
    new_stats = os.path.join(os.getcwd(), "envstats", f"{model_name}.pkl")

    # 2) BENCHMARK
    print(f"\n→ [2/3] Benchmarking '{model_name}' for {episodes} episodes …")
    bench_results = benchmark_PPO(
        bench_model_name = f'{model_name}',
        stats_file       = new_stats,
        wrapper_kwargs   = {"max_steps": bench_max_steps},
        n_episodes       = episodes,
        env_kwargs=bench_env_kwargs,
        gym_kwargs=bench_gym_kwargs
        # no callback here
    )

    # 3) ANALYZE
    run_folder = find_latest_run_folder("Benchmark", model_name)

    print(f"\n→ [3/3] Analyzing '{model_name}' …")

    analysis_results = analyze(
        model_name = run_folder,
        root_dir   = "Benchmark",
    )
    if plot_trajectories:
        animate_all_chasers(
            type_folder="Benchmark",
            name_model=run_folder,
            half_angle_deg=plot_half_angle_deg,
            cone_height=plot_cone_height,
            only_initial=False,
            save_animation=False,
            save_path = 'testAnimation2.gif'
        )
    return analysis_results

def load_config(path: str) -> Dict[str,Any]:
    """
    Loads the YAML, maps wrapper_class & activation_fn,
    turns off plotting (for batch runs), and returns cfg.
    """
    cfg = yaml.safe_load(open(path))
    # your existing CLASS_MAP logic:
    for block in ("train_env_kwargs","bench_env_kwargs"):
        name = cfg[block].pop("wrapper_class")
        cfg[block]["wrapper_class"] = CLASS_MAP[name]
    act_name = cfg["policy_dict"].pop("activation_fn")
    cfg["policy_dict"]["activation_fn"] = CLASS_MAP[act_name]
    cfg["plot_trajectories"] = False
    return cfg

def run_pipeline(cfg: Dict[str,Any]) -> Dict[str,Any]:
    """
    Unpacks cfg into train_benchmark_analyze and returns its output.
    """
    return train_benchmark_analyze(**cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c","--config", default=os.path.join(os.getcwd(), 'Configs','base_config.yaml'))
    args = p.parse_args()
    cfg = load_config(args.config)
    results = run_pipeline(cfg)
    print(results)
if __name__ == "__main__":
    main()