from gym_env.reward_shaping import reward_formulation
from old_visualizer.visualizer_close import write2text
#write2text(chaser, data_dir, file_name, step)
import numpy as np
#import cupy as np
import pdb
import gymnasium as gym
from gymnasium import spaces
import glob,os
from gym_env.dynamics import chaser_continous
import math

class ARPOD_GYM(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self,model_name, phase,benchmark=False):
        super(ARPOD_GYM, self).__init__()
        root = "Benchmark" if benchmark else "Train"
        cwd = os.getcwd()
        self.phase = phase

        runs_base = os.path.join(cwd, root, "runs")
        os.makedirs(runs_base, exist_ok=True)

        # 3) auto-increment a run ID for this model_name
        #    look for existing folders like "foo_1", "foo_2", …
        pattern = os.path.join(runs_base, f"{model_name}_*")
        existing = glob.glob(pattern)
        # extract numeric suffixes
        idxs = []
        for path in existing:
            name = os.path.basename(path)
            parts = name.split("_")
            if parts[-1].isdigit():
                idxs.append(int(parts[-1]))
        run_id = max(idxs) + 1 if idxs else 1

        # 4) construct your per-run folder names
        run_folder = f"{model_name}_{run_id}"
        save_root = os.path.join(runs_base, run_folder)
        vel_root  = os.path.join(cwd, root, "velocities",  run_folder)
        act_root  = os.path.join(cwd, root, "actuations",  run_folder)
 

        # 5) make them
        for d in (save_root, vel_root, act_root):
            os.makedirs(d, exist_ok=True)

        # 6) store for write_data()
        self.runs_dir = save_root
        self.vel_dir  = vel_root
        self.act_dir  = act_root

        print(f"[ARPOD_GYM] logging into {run_folder} (id={run_id}) under {root}/") 

    
        #self.num_envs = 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Box(low=-100.0, high=100.0,
        #                                     shape=(3,), dtype=np.float32)
        
        self.action_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(3,), dtype=np.float64)
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=-2000.0, high=2000.0,
        #                                     shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5000.0, high=5000.0,
                                            shape=(6,), dtype=np.float64)
                                            

        self.episode = 0
        #self.r_form = reward_formulation(chaser)
        self.chaser = chaser_continous(use_vbar = True, use_rbar = False, phase = self.phase)
        self.r_form = reward_formulation(self.chaser)
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'last_u' : np.array([]),
                     'inital_state' : self.chaser.state,
                     'fuel consumed' : 0 }

        self.reward_info = {'penality_smooth' : 0.001,
                    'penality_actuation' : 0.001,
                    'penality_state' : 0.001,
                    'penality_soft' : 0.001,
                    'penality_collision' : 0.001,
                    'reward_soft' : 0.001,
                    'reward_docked' : 0.001,
                    'net_reward' : 0.001,
                    'gross_reward' : 0.001}
        self.iscontinous = True

    def step(self, action):
        reward = 0
        done = False
        truncated = False
        terminated = False
        is_lower_bound = np.greater_equal(action, [-100, -100, -100])
        is_upper_bound = np.less_equal(action, [100, 100, 100])
        assert np.all(is_lower_bound) and np.all(is_upper_bound)
        #print(action)
        #rescaled_action = -10.0 + (0.5 * (action + 1.0) * (10.0 - (-10.0)) )
        #action *= 10.0
        #action = np.clip(action, -10.0, 10.0, dtype=np.float64)
        obs = self.chaser.get_next(action)
        self.chaser.update_u(action)
        self.chaser.update_state(obs)
        self.info['last_u'] = action.copy()
        actuation_fuel = np.linalg.norm(action)
        self.info['fuel consumed'] = self.info['fuel consumed'] + actuation_fuel
	    #checking collision / phase 3 terminal constraints
        p, terminated = self.r_form.terminal_conditions()
        reward += p

        if terminated:
            return obs, reward, terminated, truncated, self.info
        
        p = self.r_form.soft_penalities()
        reward += p
        
        p, truncated = self.r_form.truncate_conditions()
        reward += p

        if truncated:
            return obs, reward, terminated, truncated, self.info

        r, truncated = self.r_form.win_conditions()
        reward += r
        
        if truncated:
            return obs, reward, terminated, truncated, self.info


        r = self.r_form.soft_rewards()
        reward += r
        
        self.reward_info['penality_smooth'] = self.reward_info['penality_smooth'] + self.r_form.penality_smooth
        self.reward_info['penality_actuation'] = self.reward_info['penality_actuation'] + self.r_form.penality_actuation
        self.reward_info['penality_state'] = self.reward_info['penality_state'] + self.r_form.penality_state
        self.reward_info['penality_soft'] = self.reward_info['penality_soft'] + self.r_form.penality_soft
        self.reward_info['penality_collision'] = self.reward_info['penality_collision'] + self.r_form.penality_collision
        self.reward_info['reward_soft'] = self.reward_info['reward_soft'] + self.r_form.reward_soft
        self.reward_info['reward_docked'] = self.reward_info['reward_docked'] + self.r_form.reward_docked

        self.reward_info['net_reward'] = self.reward_info['net_reward'] + self.r_form.reward_docked + self.r_form.reward_soft - self.r_form.penality_smooth - self.r_form.penality_actuation - self.r_form.penality_state + self.r_form.penality_soft + self.r_form.penality_collision
        self.reward_info['gross_reward'] = self.reward_info['gross_reward'] + math.fabs(self.reward_info['penality_smooth']) + math.fabs(self.reward_info['penality_actuation']) + math.fabs(self.reward_info['penality_state']) + math.fabs(self.reward_info['penality_collision']) + math.fabs(self.reward_info['reward_soft']) + math.fabs(self.reward_info['reward_docked'])


        self.info['time in los'] = self.info['time in los'] + self.r_form.time_inlos
        #self.info['time in validslowzone'] = self.info['time in validslowzone'] + self.r_form.time_slowzone
        #self.info['time in los and slowzone'] = self.info['time in los and slowzone'] + self.r_form.time_inlos_slowzone
        #self.info['time in los and phase 3'] = self.info['time in los and phase3'] + self.r_form.time_inlos_phase3
        self.info['episode time'] = self.info['episode time'] + 1
        self.r_form.reset_counts()
        #data_file_name = f'{self.runs_dir}/chaser{self.episode}.txt'
        data_file_name = f'chaser{self.episode}.txt'

        self.write_data(data_file_name)

        return obs, reward, terminated, truncated, self.info

    def reset(self, seed=None,**kwargs):
        super().reset(seed=seed,**kwargs)
        print('resetting and showing info')
        print("Info")
        print("----------------")
        print("\n".join("{}\t{}".format(k, v) for k, v in self.info.items()))
        #print(self.info)
        print("----------------")
        print("Reward info")
        print("----------------")
        #print(self.reward_info)
        print("\n".join("{}\t{}".format(k, v) for k, v in self.reward_info.items()))
        print("----------------")
        print("Reward percentage")
        print("----------------")
        print(f'percentage of smooth actuation penality : {100 * self.reward_info["penality_smooth"]/self.reward_info["gross_reward"] + 0.001}')
        print(f'percentage of actuation penality : {100 * self.reward_info["penality_actuation"]/self.reward_info["gross_reward"] + 0.001}')
        print(f'percentage of state penality : {100 * self.reward_info["penality_state"]/self.reward_info["gross_reward"]+ 0.001}')
        print(f'percentage of soft penality : {-100 * self.reward_info["penality_soft"]/self.reward_info["gross_reward"] + 0.001}')
        print(f'percentage of collision penality : {-100 * self.reward_info["penality_collision"]/self.reward_info["gross_reward"]+ 0.001}')
        print(f'percentage of soft reward : {100 * self.reward_info["reward_soft"]/self.reward_info["gross_reward"]+ 0.001}')
        print(f'percentage of docked reward : {100 * self.reward_info["reward_docked"]/self.reward_info["gross_reward"] + 0.001}')
        print("----------------")

        self.chaser.reset()
        self.chaser.update_state(self.chaser.state)
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'last_u' : np.array([]),
                     'inital_state' : self.chaser.state,
                     'fuel consumed' : 0 }
                     
        #self.r_form.reset_checkpoints()

        self.reward_info = {'penality_smooth' : 0.001,
                    'penality_actuation' : 0.001,
                    'penality_state' : 0.001,
                    'penality_soft' : 0.001,
                    'penality_collision' : 0.001,
                    'reward_soft' : 0.001,
                    'reward_docked' : 0.001,
                    'net_reward' : 0.001,
                    'gross_reward' : 0.001}

        print('reseting environment')
        observation = np.array(self.chaser.state, copy=True)
        self.episode += 1
        return observation, self.info

    def render(self, mode="human"):
        pass

    def write_data(self, file_name):
        #print('writing data to text')
        #print(f'current step {self.chaser.current_step}')
        # write2text(self.chaser, 'runs', file_name, self.chaser.current_step,opt_id = 0)
        # write2text(self.chaser, 'velocities', file_name, self.chaser.current_step,opt_id = 1)
        # write2text(self.chaser, 'actuations', file_name, self.chaser.current_step,opt_id = 2)

        write2text(self.chaser, self.runs_dir, file_name, self.chaser.current_step,opt_id = 0)
        write2text(self.chaser, self.vel_dir, file_name, self.chaser.current_step,opt_id = 1)
        write2text(self.chaser, self.act_dir, file_name, self.chaser.current_step,opt_id = 2)

