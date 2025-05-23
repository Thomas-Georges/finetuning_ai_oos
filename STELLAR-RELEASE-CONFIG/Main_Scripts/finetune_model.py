import numpy as np
import os,sys
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from gym_env.environment import ARPOD_GYM
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.logger import HParam
import gymnasium as gym
import os
from torch.nn import GELU
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from typing import Callable



class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "use_sde": self.model.use_sde,
            "sde_sample_freq": self.model.sde_sample_freq,
            "target_kl": self.model.target_kl,
            "n_steps": self.model.n_steps,
            "ent_coef": self.model.ent_coef,
            "batch_size": self.model.batch_size,
            "gae_lambda": self.model.gae_lambda,
            "normalize_advantage": self.model.normalize_advantage,
            "max_grad_norm": self.model.max_grad_norm,
            "clip_range": self.model.clip_range,
            "n_epochs": self.model.n_epochs
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict2 = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
            "train/entropy_loss": 0.0,
            "train/policy_gradient_loss": 0.0,
            "train/approx_kl": 0.0,
            "train/clip_fraction": 0.0,
            "train/clip_range": 0.0,
            "train/n_updates": 0,
            "train/learning_rate": 0.0,
            "train/std": 0.0,
            "train/loss": 0.0,
            "train/explained_variance": 0.0
        }
        metric_dict = {"rollout/ep_len_mean": 0,
                        "rollout/ep_rew_mean": 0}
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=2500):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode      over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

BASE_STATS = os.path.join("Fixed_Base_Envstats", "vbar-agentv3-3.pkl")
ENV_KWARGS = {"wrapper_class": TimeLimitWrapper, "n_envs": 30}
VECENV_KWARGS ={"benchmark": False}
POLICY_DICT = {'activation_fn': GELU, 'net_arch': {'pi': [130, 44, 30], 'vf': [145, 25, 5]}, 'full_std': False, 
               'squash_output': False, 'log_std_init': -1, 'ortho_init': True, 'use_expln' : True}

CHECKBACK = CheckpointCallback(
  save_freq=2500,
  save_path="./logs/",
  name_prefix="chaser2_arpod",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

PPO_KWARGS = {
    "gamma":             0.998,
    "gae_lambda":        0.98,
    "learning_rate":     0.015,
    "n_steps":           500,
    "ent_coef":          0.0002,
    "clip_range":        0.2,
    "target_kl":         0.002,
    "batch_size":        1000,
    "verbose":           1,
    "n_epochs":          20,
    "normalize_advantage": True,
    "max_grad_norm":     0.2,
    "use_sde":           True,
    "tensorboard_log":   "arpod-ppo",
    "sde_sample_freq":   8,
    "_init_setup_model": True,
    "device":            "cuda",
}


def build_env(
    model_name: str,
    stats_file: str,
    training = True,
    make_env_kwargs= ENV_KWARGS,
    wrapper_kwargs={"max_steps": 2500},
    gym_kwargs  = VECENV_KWARGS):

    # ensure model_name is set
    gym_kwargs = {**gym_kwargs, "model_name": model_name}


    # build and normalize
    env  = make_vec_env(ARPOD_GYM, **make_env_kwargs, env_kwargs=gym_kwargs, wrapper_kwargs=wrapper_kwargs)

    if stats_file is not False:
        norm_env = VecNormalize.load(stats_file, env)
        norm_env.training = training
    else:
        norm_env = VecNormalize(env,
                            training=training,
                            norm_obs=True,
                            norm_reward=True)
    return norm_env


def train_PPO(
    model_name,
    stats_file,
    total_timesteps,
    load_model = False,
    callback = CHECKBACK,
    policy_dict = POLICY_DICT,
    wrapper_kwargs = {"max_steps": 2500},
    ppo_kwargs = PPO_KWARGS,
    env_kwargs = ENV_KWARGS,
    gym_kwargs = VECENV_KWARGS
):
    # build your training env

    train_env = build_env(
        model_name,
        stats_file,
        training = True,
        wrapper_kwargs=wrapper_kwargs,
        make_env_kwargs=env_kwargs,
        gym_kwargs=gym_kwargs,
    )

    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # find existing files like "model_name_1.zip", "model_name_2.zip", …
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

    if load_model is not False:
        model_fp = os.path.join(os.getcwd(), load_model)
        model = PPO.load(model_fp, env=train_env) 
        model.policy.action_space = train_env.action_space

    else:
        model = PPO(ActorCriticPolicy, env = train_env, **ppo_kwargs)
        model.policy.policy_kwargs = policy_dict
        model.policy.action_space = train_env.action_space

    # train
    try:

        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=True,
            progress_bar=True,
            callback=callback,
            tb_log_name=model_name,
        )

        model.save(os.path.join(os.getcwd(), 'models', f'{model_name}.zip'))
        new_stats = model.get_vec_normalize_env()  
        new_stats.save(os.path.join(os.getcwd(),'envstats', f'{model_name}.pkl'))
        return model
    finally:
        train_env.close

#test_train_model = train_PPO('vbar-agentFirstNoLoad',BASE_STATS, 50000,load_model = None)



 