import numpy as np

prices_hiv = 0.5*np.random.rand(6)
prices_cartpole = 0.5*np.random.rand(4)

player_alpha = 0
observer_alpha = 1

def reward_regularizer_hiv(rew):
    reg_rew = (1e-4) * (rew - (2e4))
    return reg_rew

def reward_regularizer_cartpole(rew):
    return rew

def player_reward_shaping_hiv(reward,obs_act):
    total_cost = obs_act @ prices_hiv
    shaped_reward = reward_regularizer_hiv(reward) - player_alpha * (total_cost)
    return shaped_reward

def observer_reward_shaping_hiv(reward,obs_act):
    total_cost = obs_act @ prices_hiv
    shaped_reward = reward_regularizer_hiv(reward) - observer_alpha * (total_cost)
    return shaped_reward

def player_reward_shaping_cartpole(reward,obs_act):
    total_cost = obs_act @ prices_cartpole
    shaped_reward = reward_regularizer_cartpole(reward) - player_alpha * (total_cost)
    return shaped_reward

def observer_reward_shaping_cartpole(reward,obs_act):
    total_cost = obs_act @ prices_cartpole
    shaped_reward = reward_regularizer_cartpole(reward) - observer_alpha * (total_cost)
    return shaped_reward
