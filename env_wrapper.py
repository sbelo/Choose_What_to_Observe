from gym import Wrapper
import numpy as np
from gym.spaces import MultiBinary
from gym.spaces import Discrete

class CHECK_:
    def __init__(self,data):
        self.data = data

class PLAYER_ENV(Wrapper):
    def __init__(self,env,budget=np.inf,prices=None,serial=False,ser_observ_space=None):
        self.serial = serial
        self.budget_orig = budget
        self.budget = budget
        self.prices = prices
        super().__init__(env)
        ss = 1
        for aa in self.observation_space.shape:
            ss *= aa
        self.dim_act = ss
        self.observation_space = ser_observ_space

    def step(self, action):
        stat = super().step(action)
        stat = list(stat)
        if self.budget <= 0:
            stat[0] = 0*stat[0]
        return stat

    def update_budget(self,action):
        if self.budget > 0:
            self.budget -= action @ self.prices

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self.prices is None:
            self.prices = 0 * obs
        self.budget = self.budget_orig
        self._last_masked = 0 * obs
        return np.zeros(self.dim_act)

    def set_model(self,observer):
        self.observer = observer



class OBSERVER_ENV(Wrapper):
    def __init__(self, env, budget=np.inf, prices=None,serial=False,ser_observ_space=None):
        self.budget_orig = budget
        self.budget = budget
        self.prices = prices
        super().__init__(env)
        self._last_obs = None
        self.last_action = None

        ss = 1
        for aa in self.observation_space.shape:
            ss *= aa
        self.dim_act = ss
        self.serial = serial
        self.observation_space = ser_observ_space
        if serial:
            self.ser_done = False
            self.ser_counter = self.dim_act
            self.ser_curr_mask = np.zeros(self.dim_act)
            self.action_space = Discrete(2)
        else:
            self.action_space = MultiBinary(self.dim_act)

    def step(self, action):
        assert self._last_obs is not None, "self._last_obs is None!"
        r_action = action
        if self.budget > 0:
            if self.serial:
                self.ser_curr_mask[self.dim_act-self.ser_counter] = action
                self.ser_counter -= 1
                r_action = self.ser_curr_mask.copy()
                self.last_action = r_action
                if self.ser_counter == 0:
                    self.ser_counter = self.dim_act
                    self.ser_curr_mask = np.zeros(self.dim_act)
                    self.ser_done = True
                else:
                    return CHECK_(np.concatenate([np.reshape(r_action,self._last_obs.shape),self._last_masked_obs]))
            else:
                self.last_action = action.copy()
            masked_obs = np.multiply(np.reshape(r_action,self._last_obs.shape),self._last_obs)
            self._last_masked_obs = masked_obs.copy()
            self.budget -= r_action @ self.prices
        else:
            masked_obs = 0*self._last_obs
            r_action = np.zeros(self.dim_act)
            if self.serial:
                self.last_action = np.zeros(self.dim_act)
                self.ser_done = True

        if self.serial:
            masked_obs = np.concatenate([np.zeros(r_action.shape),masked_obs])
        else:
            masked_obs = np.concatenate([r_action, masked_obs])
        stat = CHECK_(masked_obs)
        return stat

    def player_step(self,action):
        if self.serial:
            if self.ser_done:
                self.ser_done = False
                stat = super().step(action)
                stat = list(stat)
                stat[0] = np.concatenate([np.zeros(self._last_obs.shape),stat[0]])
                self._last_obs = stat[0][self.dim_act:]
            else:
                return np.concatenate([np.reshape(self.ser_curr_mask, self._last_obs.shape), self._last_masked_obs]), 0.0, self._last_done, {}
        else:
            stat = super().step(action)
            stat = list(stat)
            self._last_obs = stat[0]
            stat[0] = np.concatenate([np.zeros(self._last_obs.shape), stat[0]])
        self._last_done = stat[2]
        return stat

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.last_action = None
        self._last_done = False
        if self.prices is None:
            self.prices = 0 * obs
        self._last_obs = np.zeros(obs.shape)
        self.budget = self.budget_orig
        if self.serial:
            self._last_masked_obs = np.zeros(self.dim_act)
            self._last_obs = np.zeros(self.dim_act)
            self.ser_counter = self.dim_act
        return np.zeros(2*self.dim_act)

    def set_last_obs(self,obs):
        self._last_obs = obs

    def set_model(self,player):
        self.player = player



# def evaluate_policies(env,player,observer)
def evaluate_policy(player, observer, env, n_eval_episodes = 10, deterministic = True, render = False, return_episode_rewards = False, serial = False,budget = None, prices = None, reward_shaping = None):
    shaped_episode_rewards, episode_rewards, episode_lengths = [], [], []
    finished = False
    obs = env.reset()
    masks = 0 * obs
    evals_left = n_eval_episodes
    masked_obs = np.multiply(masks,obs)
    dones, p_states, o_states = np.zeros(env.num_envs,dtype=bool), None, None
    shaped_episode_reward = np.zeros(env.num_envs)
    episode_reward = np.zeros(env.num_envs)
    episode_length = np.zeros(env.num_envs)
    if prices is None:
        prices = np.zeros(obs.shape[1])
    if budget is None:
        budget = np.inf
    curr_budget = budget*np.ones(env.num_envs)
    mask_hist = None
    while True:
        if serial:
            masks = np.zeros(masks.shape)
            for dd in range(masks.shape[1]):
                masks[:,dd] = observer.i_predict(np.concatenate([masks,masked_obs],axis=1), dones)# state=o_states, mask=dones, deterministic=deterministic)
        else:
            masks = observer.i_predict(np.concatenate([masks,masked_obs],axis=1), dones)#state=o_states, mask=dones, deterministic=deterministic)
        masks_tmp = np.zeros(masks.shape)
        masks_tmp[curr_budget > 0,:] = masks[curr_budget > 0,:]
        masks = masks_tmp
        curr_budget -= prices @ masks.T

        if mask_hist is None:
            mask_hist = masks
        else:
            mask_hist += masks
        masked_obs = np.multiply(masks,obs)
        actions = player.i_predict(np.concatenate([masks,masked_obs],axis=1), dones) #state=p_states, mask=dones, deterministic=deterministic)
        obs, rewards, dones, _infos = env.step(actions)
        episode_reward += rewards
        episode_length += np.ones(env.num_envs)
        if reward_shaping is not None:
            for ee in range(env.num_envs):
                shaped_episode_reward[ee] += reward_shaping(rewards[ee],masks[ee],prices)

        for ee in range(env.num_envs):
            if dones[ee]:
                episode_rewards.append(episode_reward[ee])
                if reward_shaping is not None:
                    shaped_episode_rewards.append(shaped_episode_reward[ee])
                    shaped_episode_reward[ee] = 0
                episode_lengths.append(episode_length[ee])
                episode_reward[ee] = 0
                episode_length[ee] = 0
                evals_left -= 1
                masks[ee,:] = 0 * masks[ee,:]
                masked_obs[ee,:] = 0 * masked_obs[ee,:]
                curr_budget[ee] = budget
                if evals_left <= 0:
                    finished = True
                    break
        if finished:
            break
        if render:
            env.render()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    mean_shaped_reward = np.mean(shaped_episode_rewards)
    std_shaped_reward = np.std(shaped_episode_rewards)
    mask_hist = np.sum(mask_hist,axis=0)
    mask_hist = (1.0/n_eval_episodes)*mask_hist
    if return_episode_rewards:
        if reward_shaping is not None:
            return shaped_episode_rewards, episode_rewards, episode_lengths
        else:
            return episode_rewards, episode_lengths

    if reward_shaping is not None:
        return mean_shaped_reward, std_shaped_reward, mean_reward, std_reward
    else:
        return mean_reward, std_reward
