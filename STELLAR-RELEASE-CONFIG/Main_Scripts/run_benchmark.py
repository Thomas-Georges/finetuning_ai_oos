import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



from stable_baselines3 import PPO
from finetune_model import build_env, TimeLimitWrapper
from stable_baselines3.common.evaluation import evaluate_policy


cwd = os.getcwd()
BASE_STATS = os.path.join(cwd, "envstats","vbar-agentFirstNoLoad.pkl")
ENV_KWARGS = {"wrapper_class": TimeLimitWrapper, "n_envs": 1}
VECENV_KWARGS ={"benchmark": True}

def benchmark_PPO(
    bench_model_name,
    stats_file,
    wrapper_kwargs={"max_steps": 2500},
    env_kwargs = ENV_KWARGS,
    gym_kwargs = VECENV_KWARGS,
    n_episodes = 10,
):
    # build your eval env (benchmark mode => no further norm updates)
    bench_env = build_env(
        bench_model_name,
        stats_file=stats_file,
        training = False,
        make_env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        gym_kwargs = gym_kwargs,
    )

    model_fp = os.path.join(os.getcwd(), 'models',f"{bench_model_name}.zip")

    try:

        model = PPO.load(model_fp, env=bench_env, force_reset=True)

        terminal_data, fuel_data = evaluate_policy(
        model = model,env = bench_env, n_eval_episodes=n_episodes, deterministic=False
    )
        return {
            "terminal": terminal_data,
            "fuel":     fuel_data,
            # "length":   eplen_data,
            # "runtime":  runtime,
            # "test1": test1,
            # "test2": test2
            }
    finally:
        bench_env.close()


#first_bench = benchmark_PPO('vbar-agentFirstNoLoad', stats_file=BASE_STATS)

#bench_env = build_env(model_name='vbar-agentFirst', stats_file=BASE_STATS, make_env_kwargs=ENV_KWARGS)
# bench_env = build_env(model_name='vbar-agentSecond', stats_file=BASE_STATS, make_env_kwargs=ENV_KWARGS)

# modelname = 'vbar-agentSecond'
# model_dir = f'models/{modelname}'

# loaded_model = PPO.load(model_dir, env=bench_env, print_system_info=True, force_reset=True)
# evaluate_policy(model=loaded_model, env=bench_env, n_eval_episodes=100, deterministic=False)
