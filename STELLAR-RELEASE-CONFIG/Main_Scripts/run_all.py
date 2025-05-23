import os
import json
import argparse
import glob
from typing import Dict, Any

from finetune_model               import train_PPO, POLICY_DICT as DEFAULT_POLICY_DICT,ENV_KWARGS as DEFAULT_TRAIN_ENV_KWARGS, VECENV_KWARGS as DEFAULT_TRAIN_GYM_KWARGS, BASE_STATS, CHECKBACK,PPO_KWARGS as DEFAULT_PPO_KWARGS
from run_benchmark                import benchmark_PPO, ENV_KWARGS as DEFAULT_BENCH_ENV_KWARGS, VECENV_KWARGS as DEFAULT_BENCH_GYM_KWARGS
from compute_model_stats       import analyze
from animate_chaser  import animate_all_chasers    # ← import here

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
    train_timesteps: int,
    benchmark_episodes: int = 100,
    train_max_steps: int    = 2500,
    bench_max_steps: int    = 2500,
    policy_dict = DEFAULT_POLICY_DICT,
    ppo_kwargs: Dict[str,Any]       = DEFAULT_PPO_KWARGS,
    train_env_kwargs: Dict[str,Any] = DEFAULT_TRAIN_ENV_KWARGS,
    train_gym_kwargs: Dict[str,Any] = DEFAULT_TRAIN_GYM_KWARGS,
    bench_env_kwargs: Dict[str,Any] = DEFAULT_BENCH_ENV_KWARGS,
    bench_gym_kwargs: Dict[str,Any] = DEFAULT_BENCH_GYM_KWARGS,
    callback                        = CHECKBACK,
    plot_trajectories = True,
    plot_half_angle_deg: float = 20,
    plot_cone_height: float    = 200,
) -> Dict[str, Any]:
    """
    1) Train   (with `callback`)
    2) Benchmark
    3) Analyze
    """
    ppo_kwargs       = ppo_kwargs       or {}
    train_env_kwargs = train_env_kwargs or DEFAULT_TRAIN_ENV_KWARGS
    train_gym_kwargs = train_gym_kwargs or DEFAULT_TRAIN_GYM_KWARGS
    bench_env_kwargs = bench_env_kwargs or DEFAULT_BENCH_ENV_KWARGS
    bench_gym_kwargs = bench_gym_kwargs or DEFAULT_BENCH_GYM_KWARGS

    # 1) TRAIN
    print(f"\n→ [1/3] Training '{model_name}' for {train_timesteps} steps …")
    train_PPO(
        model_name       = model_name,
        stats_file       = BASE_STATS,
        total_timesteps  = train_timesteps,
        ppo_kwargs       = ppo_kwargs,
        env_kwargs       = train_env_kwargs,
        gym_kwargs       = train_gym_kwargs,
        wrapper_kwargs   = {"max_steps": train_max_steps},
        callback         = callback,
        policy_dict= policy_dict
    )

    # load the new normalizer
    new_stats = os.path.join(os.getcwd(), "envstats", f"{model_name}.pkl")

    # 2) BENCHMARK
    print(f"\n→ [2/3] Benchmarking '{model_name}' for {benchmark_episodes} episodes …")
    bench_results = benchmark_PPO(
        bench_model_name = model_name,
        stats_file       = new_stats,
        wrapper_kwargs   = {"max_steps": bench_max_steps},
        n_episodes       = benchmark_episodes,
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
            only_initial=False
        )
    return analysis_results

def main():
    parser = argparse.ArgumentParser(description="Train→Benchmark→Analyze")
    parser.add_argument("model_name", nargs = '?',default = 'testModel', help="Unique run name")
    parser.add_argument("timesteps",  nargs = '?',type=int,default = 100000, help="Training timesteps")
    parser.add_argument("--episodes",      type=int, default=25, help="Benchmark episodes")
    parser.add_argument("--train-max-steps", type=int, default=2500, help="Max steps/episode (train)")
    parser.add_argument("--bench-max-steps", type=int, default=2500, help="Max steps/episode (bench)")
    parser.add_argument("--ppo-kwargs",      type=json.loads, default="{}", help="JSON PPO kwargs")
    parser.add_argument("--policy-dict",      type=json.loads, default="{}", help="JSON PPO kwargs")
    parser.add_argument("--train-env-kwargs", type=json.loads, default="{}", help="JSON ENV_KWARGS override")
    parser.add_argument("--train-gym-kwargs", type=json.loads, default="{}", help="JSON GYM_KWARGS override")
    parser.add_argument("--bench-env-kwargs", type=json.loads, default="{}", help="JSON ENV_KWARGS override for bench")
    parser.add_argument("--bench-gym-kwargs", type=json.loads, default="{}", help="JSON GYM_KWARGS override for bench")

    args = parser.parse_args()
    print(args.model_name)

    results = train_benchmark_analyze(
        model_name         = args.model_name,
        train_timesteps    = args.timesteps,
        benchmark_episodes = args.episodes,
        train_max_steps    = args.train_max_steps,
        bench_max_steps    = args.bench_max_steps,
        ppo_kwargs         = args.ppo_kwargs,
        train_env_kwargs   = args.train_env_kwargs,
        train_gym_kwargs   = args.train_gym_kwargs,
        bench_env_kwargs   = args.bench_env_kwargs,
        bench_gym_kwargs   = args.bench_gym_kwargs,
        callback           = CHECKBACK,    # only for train
    )

    print(results)

    print("\n=== Final Summary ===")
    # print("Benchmark:", results["benchmark"])
    # print("Analysis: ", results["analysis"])
    print("\n✅ Done.")

if __name__ == "__main__":
    main()