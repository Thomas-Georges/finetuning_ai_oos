import argparse, copy,os, pandas as pd
from run_all_configs import load_config, run_pipeline

def sweep_param(config_path: str, param: str, values: list, out_csv: str):
    base = load_config(config_path)
    list_keys = []
    list_concat = []
    for v in values:
        print(v)
        cfg = copy.deepcopy(base)
        # support nested keys, e.g. "ppo_kwargs.learning_rate"
        sub = cfg
        *keys, last = param.split(".")
        for k in keys:
            sub = sub.setdefault(k, {})
        sub[last] = v

        print(f"\n--- Running with {param}={v} ---")
        out = run_pipeline(cfg)
        new_df = pd.DataFrame(out)
        list_keys.append(f'{param}_{v}')
        list_concat.append(new_df)

    df = pd.concat(list_concat, axis=1, keys = list_keys)
    df.to_csv(out_csv, index = True)
    print("Saved sweep to", out_csv)
    return df

sweep_param(os.path.join(os.getcwd(), 'Configs/base_config.yaml'), 'ppo_kwargs.max_grad_norm', [0.1,0.2,0.5,1], os.path.join(os.getcwd(), 'testSweep.csv'))
# if __name__=="__main__":
#     p = argparse.ArgumentParser("Hyperparam sweep")
#     p.add_argument("--config","-c", default="base_config.yaml")
#     p.add_argument("param", help="e.g. ppo_kwargs.learning_rate")
#     p.add_argument("values", help="comma-sep values, e.g. 0.001,0.005,0.01")
#     p.add_argument("--out", default="sweep.csv")
#     args = p.parse_args()

#     def cast(x):
#         if x.isdigit(): return int(x)
#         try: return float(x)
#         except: return x

#     vals = [cast(v) for v in args.values.split(",")]
#     sweep_param(args.config, args.param, vals, args.out)