
if __name__ == "__main__":
    # import gym
    #
    # from stable_baselines import PPO2
    # import stable_baselines.common.vec_env.subproc_vec_env as venv
    #
    # from env_wrapper import *
    # from agents_wrapper import *
    # from subproc_vec_env import *
    # # env = gym.make('CartPole-v1')
    # env = venv.SubprocVecEnv([lambda : gym.make('CartPole-v1') for _ in range(4)])
    # model = PPO2('MlpLstmPolicy', env, verbose=1,nminibatches=2)
    # obs = env.reset()
    # model.predict(obs)
    # model.learn(total_timesteps=10000)


    # import gym
    import whynot as wn
    import os.path
    from stable_baselines import PPO2
    import stable_baselines.common.vec_env.subproc_vec_env as org_sub_vec
    from env_wrapper import *
    from agents_wrapper import *
    from subproc_vec_env import *
    from gym.spaces.box import Box
    import wandb
    from wandb.tensorflow import WandbHook

    wandb.init(project="choose_what_to_observe", sync_tensorboard=True, name=None)
    ######################################################################
    # Environments:
    # HIV:
    # work_env = wn.gym
    # env_name = 'HIV-v0'
    # state_space_low = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # state_space_high = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    # obs_space = Box(state_space_low, state_space_high, dtype=np.float64)
    ################################

    # cartpole
    work_env = gym
    env_name = 'CartPole-v1'
    state_space_low = np.asarray([0.0, 0.0, 0.0, 0.0, -4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38])
    state_space_high = np.asarray([1.0, 1.0, 1.0, 1.0, 4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38])
    obs_space = Box(state_space_low, state_space_high, dtype=np.float64)
    ######################################################################
   # Hyper parameters:
    num_cpu = 4
    nminibatches = 2
    verbose = 0
    dim_obs = int(state_space_high.size / 2)
    serial = True
    prices = None #np.asarray([25,25,40,40])#,30,30])
    learn_steps = 1000
    iters = 1000
    budget = np.inf # 2000
    n_eval_episodes = 100
    player_policy = 'MlpLstmPolicy'
    observer_policy = 'MlpLstmPolicy'
    ######################################################################
    # budget_constraint = None
    # define reward constraints for budget:
    alpha = 0
    def budget_constraint(reward,actions,prices):
        total_cost = actions @ prices
        constraint_reward = reward -alpha*(total_cost)
        return constraint_reward
    ######################################################################
    # Setup environments:
    # define fake environments for training:
    p_env = SubprocVecEnv([lambda : PLAYER_ENV(work_env.make(env_name), budget=budget, prices=prices,serial=serial,ser_observ_space=obs_space) for _ in range(num_cpu)],player_flag=True,serial=serial)
    o_env = SubprocVecEnv([lambda : OBSERVER_ENV(work_env.make(env_name), budget=budget, prices=prices, budget_constraint=budget_constraint, serial=serial,ser_observ_space=obs_space) for _ in range(num_cpu)],player_flag=False,serial=serial)

    # define test environment
    test_env = org_sub_vec.SubprocVecEnv([lambda : work_env.make(env_name) for _ in range(num_cpu)])
    ######################################################################
    # Setup agents:
    player = GENERAL_WRAPPER(raw_predict1,policy=player_policy, env=p_env, verbose=verbose,nminibatches=nminibatches)
    observer = GENERAL_WRAPPER(raw_predict1,policy=observer_policy, env=o_env, verbose=verbose,nminibatches=nminibatches)
    p_env.set_model(observer)
    o_env.set_model(player)
    ######################################################################
    # Train and evaluate:
    p_env.reset()
    o_env.reset()

    mean_reward, std_reward = evaluate_policy(player, observer, test_env, n_eval_episodes=n_eval_episodes, deterministic=True, render=False, return_episode_rewards=False, serial=serial,budget=budget,prices=prices)
    wandb.log({"mean_reward": mean_reward}, step=0)
    wandb.log({"std_reward": std_reward}, step=0)

    for iter in range(1,iters + 1):
        p_env.reset()
        o_env.reset()
        fac = 1
        if serial:
            fac = dim_obs
        if verbose == 1:
            print("----------------------------------------------------------------------------")
            print("Observer learns")
            print("----------------------------------------------------------------------------")
        observer.learn(total_timesteps=learn_steps*fac)
        p_env.set_model(observer)
        if verbose == 1:
            print("----------------------------------------------------------------------------")
            print("Player learns")
            print("----------------------------------------------------------------------------")
        player.learn(total_timesteps=learn_steps)
        o_env.set_model(player)

        if verbose == 1:
            print("----------------------------------------------------------------------------")
        # evaluate:
        mean_reward, std_reward = evaluate_policy(player, observer, test_env, n_eval_episodes=n_eval_episodes, deterministic=True, render=False, return_episode_rewards=False, serial=serial, budget=budget,prices=prices)
        print("Evaluation at " + str(iter) + " epochs: mean reward: " + str(mean_reward) + ", std reward: " + str(std_reward))
        wandb.log({"mean_reward": mean_reward}, step=iter)
        wandb.log({"std_reward": std_reward}, step=iter)
    # Save models to wandb
    # player.save(os.path.join(wandb.run.dir, "player.ckpt"))
    # observer.save(os.path.join(wandb.run.dir, "observer.ckpt"))
