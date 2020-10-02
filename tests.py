
if __name__ == "__main__":
    # import gym
    import whynot as wn
    from stable_baselines.bench.monitor import Monitor
    import os.path
    from stable_baselines import PPO2
    import stable_baselines.common.vec_env.subproc_vec_env as org_sub_vec
    from env_wrapper import *
    from agents_wrapper import *
    from subproc_vec_env import *
    from gym.spaces.box import Box
    import wandb

    wandb.init(project="choose_what_to_observe", sync_tensorboard=True, name=None)
    ######################################################################
    # Environments:
    # HIV:
    work_env = wn.gym
    env_name = 'HIV-v0'
    state_space_low = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state_space_high = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    obs_space = Box(state_space_low, state_space_high, dtype=np.float64)
    ################################

    # cartpole
    # work_env = gym
    # env_name = 'CartPole-v1'
    # state_space_low = np.asarray([0.0, 0.0, 0.0, 0.0, -4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38])
    # state_space_high = np.asarray([1.0, 1.0, 1.0, 1.0, 4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38])
    # obs_space = Box(state_space_low, state_space_high, dtype=np.float64)
    ######################################################################
   # Hyper parameters:
    num_cpu = 4
    nminibatches = 1
    verbose = 0
    dim_obs = int(state_space_high.size / 2)
    serial = True
    prices = 0.5*np.random.rand(6)
    episode_length = 400
    if serial:
        episode_length_obs = dim_obs*episode_length
    else:
        episode_length_obs = episode_length

    learn_steps = 1*num_cpu*episode_length
    initial_player_learn_steps = 1*num_cpu*episode_length
    # initial_observer_learn_steps = 20*num_cpu*episode_length
    iters = 40000
    budget = np.inf
    n_eval_episodes = 20
    player_policy = 'MlpLstmPolicy'
    observer_policy = 'MlpLstmPolicy'
    monitor = True
    ######################################################################
    # budget_constraint = None
    # define reward constraints for budget:
    def reward_regularizer(rew):
        reg_rew = (1e-4) * (rew - (2e4) * np.ones(rew.shape))
        return reg_rew


    alpha = 1
    def reward_shaping_obs(reward,actions,prices):
        total_cost = actions @ prices
        shaped_reward = reward_regularizer(reward) -alpha*(total_cost)
        return shaped_reward

    beta = 0
    def reward_shaping_ply(reward,actions,prices):
        total_cost = actions @ prices
        shaped_reward = reward_regularizer(reward) - beta *(total_cost)
        return shaped_reward

    reward_shaping = reward_shaping_obs
    ######################################################################
    # Setup environments:
    # define fake environments for training:
    p_env = SubprocVecEnv1([lambda : PLAYER_ENV(work_env.make(env_name), budget=budget, prices=prices,serial=serial,ser_observ_space=obs_space) for _ in range(num_cpu)],player_flag=True,serial=serial,reward_shaping=reward_shaping_ply,prices=prices,monitor=monitor)
    o_env = SubprocVecEnv1([lambda : OBSERVER_ENV(work_env.make(env_name), budget=budget, prices=prices, serial=serial,ser_observ_space=obs_space) for _ in range(num_cpu)],player_flag=False,serial=serial,reward_shaping=reward_shaping_obs,prices=prices,monitor=monitor)
    # define test environment
    test_env = org_sub_vec.SubprocVecEnv([lambda : work_env.make(env_name) for _ in range(num_cpu)])
    ######################################################################
    # Setup agents:
    player = GENERAL_WRAPPER(raw_predict1,policy=player_policy, env=p_env, verbose=verbose,nminibatches=nminibatches,n_steps=episode_length)
    observer = GENERAL_WRAPPER(raw_predict1,policy=observer_policy, env=o_env, verbose=verbose,nminibatches=nminibatches,n_steps=episode_length_obs)
    # observer.random_on()
    # observer.fixed_on(1)
    p_env.set_model(observer)
    o_env.set_model(player)
    ######################################################################
    # Train and evaluate:
    curr_ii = 0
    p_env.reset()
    o_env.reset()
    # first, train the player with random observations:
    #####################################################################
    # for bb in range(10000):
    #     player.learn(total_timesteps=initial_player_learn_steps,reset_num_timesteps=(bb == 0))
    # # p_env.reward_shaping = reward_shaping_obs
    #     player_pre_rewards = np.asarray(p_env.get_episode_rewards())
    # # print(p_env.get_episode_rewards())
    # # print(player_pre_rewards.shape)
    # # print(player_pre_rewards)
    #     mean_player_pre_rewards = player_pre_rewards #np.mean(player_pre_rewards, axis=0)
    #     ii = 0
    #     for ii in range(len(mean_player_pre_rewards)):
    #         wandb.log({"player_train_episode_reward": mean_player_pre_rewards[ii]}, step=ii + curr_ii )
    #         wandb.log({"observer_train_episode_reward": 0}, step=ii + curr_ii)
    #     curr_ii += ii
    #     if bb % 100 == 0:
    #         print(bb)
    #     p_env.remove_history()
    # # print(np.mean(player_pre_rewards,axis=0))
    # # print("-----------------------------------------------------------")
    # # print(p_env.get_episode_lengths())
    # # print("-----------------------------------------------------------")
    # p_env.remove_history()
    ###########################################################
    # observer.random_off()
    p_env.set_model(observer)
    o_env.set_model(player)

    # observer.learn(total_timesteps=initial_observer_learn_steps)
    # o_env.remove_history()
    # p_env.set_model(observer)

    p_env.reset()
    o_env.reset()

    mean_shaped_reward, std_shaped_reward, mean_reward, std_reward = evaluate_policy(player, observer, test_env, n_eval_episodes=n_eval_episodes, deterministic=True, render=False, return_episode_rewards=False, serial=serial,budget=budget,prices=prices,reward_shaping=reward_shaping)
    print("Evaluation at 0 epochs: mean reward: " + str(mean_reward) + ", std reward: " + str(
        std_reward) + ", mean shaped reward: " + str(mean_shaped_reward) + ", std shaped reward: " + str(
        std_shaped_reward))
    # wandb.log({"mean_reward": mean_reward}, step=0)
    # wandb.log({"std_reward": std_reward}, step=0)
    # wandb.log({"mean_shaped_reward": mean_shaped_reward}, step=0)
    # wandb.log({"std_shaped_reward": std_shaped_reward}, step=0)
    o_count = 0
    p_count = 0
    for iter in range(1,iters + 1):
        # p_env.reset()
        # o_env.reset()
        fac = 1
        if serial:
            fac = dim_obs
        if verbose == 1:
            print("----------------------------------------------------------------------------")
            print("Observer learns")
            print("----------------------------------------------------------------------------")
        observer.learn(total_timesteps=learn_steps*fac,reset_num_timesteps=False)
        # observer.num_timesteps = iter*learn_steps
        # observer.fixed_on(1)
        p_env.set_model(observer)
        if verbose == 1:
            print("----------------------------------------------------------------------------")
            print("Player learns")
            print("----------------------------------------------------------------------------")
        player.learn(total_timesteps=learn_steps,reset_num_timesteps=False)
        # player.num_timesteps = iter*learn_steps #+ initial_player_learn_steps
        o_env.set_model(player)

        if verbose == 1:
            print("----------------------------------------------------------------------------")
        # evaluate:
        # if iter % 100 == 0:
        #
        #     mean_shaped_reward, std_shaped_reward, mean_reward, std_reward = evaluate_policy(player, observer, test_env, n_eval_episodes=n_eval_episodes, deterministic=True, render=False, return_episode_rewards=False, serial=serial, budget=budget,prices=prices,reward_shaping=reward_shaping)
        #     print("Evaluation at " + str(iter) + " epochs: mean reward: " + str(mean_reward) + ", std reward: " + str(std_reward) + ", mean shaped reward: " + str(mean_shaped_reward) + ", std shaped reward: " + str(std_shaped_reward))
        # wandb.log({"mean_reward": mean_reward}, step=iter)
        # wandb.log({"std_reward": std_reward}, step=iter)
        # wandb.log({"mean_shaped_reward": mean_shaped_reward}, step=iter)
        # wandb.log({"std_shaped_reward": std_shaped_reward}, step=iter)
        act_traj_o = o_env.get_actions_trajectory()

        act_traj_p = p_env.get_actions_trajectory()

        o_episode_rewards = np.asarray(o_env.get_episode_rewards())
        p_episode_rewards = np.asarray(p_env.get_episode_rewards())
        o_env.remove_history()
        p_env.remove_history()


        mean_o_episode_rewards = o_episode_rewards #np.mean(o_episode_rewards,axis=0)
        mean_p_episode_rewards = p_episode_rewards #np.mean(p_episode_rewards,axis=0)
        for r in range(min(len(mean_o_episode_rewards),len(mean_p_episode_rewards))):
            unpacked_obs_a = unpack_obs_act(act_traj_o[r],dim_obs)
            per_act_precent_o = 100.0*unpacked_obs_a.mean(axis=0)
            for oo in range(dim_obs):
                wandb.log({"observer_train_feature " + str(oo + 1) : per_act_precent_o[oo]}, step=o_count)

            wandb.log({"observer_train_feature_total": 100.0*unpacked_obs_a.mean()}, step=o_count)
            wandb.log({"observer_train_episode_reward": mean_o_episode_rewards[r]}, step=o_count)
            o_count += 1


            unpacked_pla_a = unpack_obs_act(act_traj_p[r],dim_obs)
            per_act_precent_p = 100.0*unpacked_pla_a.mean(axis=0)
            for oo in range(dim_obs):
                wandb.log({"player_train_feature " + str(oo + 1) : per_act_precent_p[oo]}, step=p_count)

            wandb.log({"player_train_feature_total": 100.0*unpacked_pla_a.mean()}, step=p_count)
            wandb.log({"player_train_episode_reward": mean_p_episode_rewards[r]}, step=p_count)
            p_count += 1

    # Save models to wandb
    # player.save(os.path.join(wandb.run.dir, "player.ckpt"))
    # observer.save(os.path.join(wandb.run.dir, "observer.ckpt"))
