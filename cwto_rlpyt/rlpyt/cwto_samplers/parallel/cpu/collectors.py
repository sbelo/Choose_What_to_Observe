
import numpy as np

from rlpyt.cwto_samplers.collectors import (DecorrelatingStartCollector,
    BaseEvalCollector)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
    buffer_method)


class CpuResetCollector(DecorrelatingStartCollector):
    """Collector which executes ``agent.step()`` in the sampling loop (i.e.
    use in CPU or serial samplers.)

    It immediately resets any environment which finishes an episode.  This is
    typically indicated by the environment returning ``done=True``.  But this
    collector defers to the ``done`` signal only after looking for
    ``env_info["traj_done"]``, so that RL episodes can end without a call to
    ``env_reset()`` (e.g. used for episodic lives in the Atari env).  The 
    agent gets reset based solely on ``done``.
    """

    mid_batch_reset = True

    def collect_batch(self, player_agent_inputs, observer_agent_inputs, player_traj_infos, observer_traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        player_agent_buf, player_env_buf = self.player_samples_np.agent, self.player_samples_np.env
        observer_agent_buf, observer_env_buf = self.observer_samples_np.agent, self.observer_samples_np.env
        player_completed_infos = list()
        observer_completed_infos = list()
        observer_observation, observer_action, observer_reward = observer_agent_inputs
        player_observation, player_action, player_reward = player_agent_inputs
        observer_obs_pyt, observer_act_pyt, observer_rew_pyt = torchify_buffer(observer_agent_inputs)
        player_obs_pyt, player_act_pyt, player_rew_pyt = torchify_buffer(player_agent_inputs)
        observer_agent_buf.prev_action[0] = np.reshape(observer_action,observer_agent_buf.prev_action[0].shape)  # Leading prev_action.
        observer_env_buf.prev_reward[0] = observer_reward
        player_agent_buf.prev_action[0] = np.reshape(player_action,player_agent_buf.prev_action[0].shape)  # Leading prev_action.
        player_env_buf.prev_reward[0] = player_reward
        self.agent.sample_mode(itr)
        observer_agent_info = [{} for _ in range(len(self.envs))]
        player_agent_info = [{} for _ in range(len(self.envs))]
        ser_count = 0
        t = 0
        prev_reset = np.zeros(len(self.envs), dtype=bool)
        player_done = np.zeros(len(self.envs),dtype=bool) 
        player_env_info = [None for _ in range(len(self.envs))] 
        observer_done = np.zeros(len(self.envs),dtype=bool) 
        observer_env_info = [None for _ in range(len(self.envs))] 
        while t < self.batch_T:
            # all envs must be in the same player_turn status!
            if self.envs[0].player_turn:
                player_env_buf.observation[t] = player_observation
                player_env_buf.reward[t] = player_reward
                for ee in range(len(self.envs)):
                    player_env_buf.done[t,ee] = player_done[ee]
                    if player_env_info[ee] is not None:
                        player_env_buf.env_info[t,ee] = player_env_info[ee]
                player_done = np.zeros(len(self.envs),dtype=bool)
                player_env_info = [None for _ in range(len(self.envs))]
                player_act_pyt, player_agent_info = self.agent.step(player_obs_pyt, player_act_pyt.float(), player_rew_pyt)
                player_action = numpify_buffer(player_act_pyt)
                for b, env in enumerate(self.envs):
                    o, r, d, env_info = env.step(player_action[b])
                    if d and (env.observer_reward_shaping is not None):
                        r_obs = env.observer_reward_shaping(r,env.last_obs_act)
                    else:
                        r_obs = r
                    observer_traj_infos[b].step(observer_observation[b], observer_action[b], r_obs, d,
                                                observer_agent_info[b], env_info)
                    if getattr(env_info, "traj_done", d):
                        observer_completed_infos.append(observer_traj_infos[b].terminate(o))
                        observer_traj_infos[b] = self.TrajInfoCls()
                        if env.player_reward_shaping is not None:
                            r_ply = env.player_reward_shaping(r, env.last_obs_act)
                        else:
                            r_ply = r
                        player_traj_infos[b].step(player_observation[b], player_action[b], r_ply, d, player_agent_info[b], env_info)
                        player_completed_infos.append(player_traj_infos[b].terminate(env.player_observation_space.null_value()))
                        player_traj_infos[b] = self.TrajInfoCls()
                        prev_reset[b] = True
                        o = env.reset()
                    if d:
                        self.agent.reset_one(idx=b)
                        player_reward[b] = r_ply
                        player_done[b] = d

                        if env_info:
                            player_env_buf.env_info[t, b] = env_info
                            player_env_info[b] = env_info

                    observer_observation[b] = o
                    observer_reward[b] = r_obs

                    observer_done[b] = d
                    if env_info:
                        observer_env_info[b] = env_info

                player_agent_buf.action[t] = player_action
                if player_agent_info:
                    player_agent_buf.agent_info[t] = player_agent_info
                t += 1
            else:
                while not self.envs[0].player_turn:
                    pturn = self.envs[0].player_turn
                    observer_env_buf.observation[ser_count] = observer_observation
                    observer_env_buf.reward[ser_count] = observer_reward
                    for ee in range(len(self.envs)):
                        observer_env_buf.done[ser_count, ee] = observer_done[ee]
                        if observer_env_info[ee] is not None:
                            observer_env_buf.env_info[ser_count, ee] = observer_env_info[ee]
                    observer_done = np.zeros(len(self.envs),dtype=bool)
                    observer_env_info = [None for _ in range(len(self.envs))]
                    observer_act_pyt, observer_agent_info = self.agent.step(observer_obs_pyt, observer_act_pyt, observer_rew_pyt)
                    observer_action = numpify_buffer(observer_act_pyt)
                    for b, env in enumerate(self.envs):
                        assert pturn == env.player_turn
                        o, r, d, env_info = env.step(observer_action[b])
                        assert not d
                        if env.player_turn:
                            if prev_reset[b]:
                                prev_reset[b] = False
                            else:
                                player_traj_infos[b].step(player_observation[b], player_action[b], r, d, player_agent_info[b], env_info)
                                player_reward[b] = r
                                player_done[b] = d
                                if env_info:
                                    player_env_info[b] = env_info

                            player_observation[b] = o
                        else:

                            observer_traj_infos[b].step(observer_observation[b], observer_action[b], r, d, observer_agent_info[b], env_info)
                            observer_observation[b] = o
                            observer_reward[b] = r
                            observer_env_buf.done[ser_count, b] = d
                            observer_done[b] = d
                            if env_info:
                                observer_env_buf.env_info[ser_count, b] = env_info
                                observer_env_info[b] = env_info

                    observer_agent_buf.action[ser_count] = observer_action
                    if observer_agent_info:
                        observer_agent_buf.agent_info[ser_count] = observer_agent_info
                    ser_count += 1

        if "bootstrap_value" in player_agent_buf:
            # agent.value() should not advance rnn state.
            player_agent_buf.bootstrap_value[:] = self.agent.value(player_obs_pyt, player_act_pyt, player_rew_pyt,is_player=True)
        if "bootstrap_value" in observer_agent_buf:
            # agent.value() should not advance rnn state.
            observer_agent_buf.bootstrap_value[:] = self.agent.value(observer_obs_pyt, observer_act_pyt, observer_rew_pyt,is_player=False)

        return AgentInputs(player_observation, player_action, player_reward), player_traj_infos, player_completed_infos, AgentInputs(observer_observation, observer_action, observer_reward), observer_traj_infos, observer_completed_infos


class CpuEvalCollector(BaseEvalCollector):
    """Offline agent evaluation collector which calls ``agent.step()`` in 
    sampling loop.  Immediately resets any environment which finishes a
    trajectory.  Stops when the max time-steps have been reached, or when
    signaled by the master process (i.e. if enough trajectories have
    completed).
    """

    def collect_evaluation(self, itr):
        observer_traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        player_traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observer_observations = list()
        player_observations = list()
        for env in self.envs:
            observer_observations.append(env.reset())
            player_observations.append(env.player_observation_space.null_value())
        observer_observation = buffer_from_example(observer_observations[0], len(self.envs))
        player_observation = buffer_from_example(player_observations[0], len(self.envs))
        observer_reward = np.zeros(len(self.envs), dtype="float32")
        player_reward = np.zeros(len(self.envs), dtype="float32")
        for b, o in enumerate(observer_observations):
            observer_observation[b] = o
        observer_action = buffer_from_example(self.envs[0].observer_action_space.null_value(),len(self.envs))
        player_action = buffer_from_example(self.envs[0].player_action_space.null_value(),len(self.envs))
        observer_obs_pyt, observer_act_pyt, observer_rew_pyt = torchify_buffer((observer_observation, observer_action, observer_reward))
        player_obs_pyt, player_act_pyt, player_rew_pyt = torchify_buffer((player_observation, player_action, player_reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        observer_agent_info = [{} for _ in range(len(self.envs))]
        player_agent_info = [{} for _ in range(len(self.envs))]
        prev_reset = np.zeros(len(self.envs), dtype=bool)
        for t in range(self.max_T):
            for _ in range(2):
                if self.envs[0].player_turn:
                    player_act_pyt, player_agent_info = self.agent.step(player_obs_pyt, player_act_pyt, player_rew_pyt)
                    player_action = numpify_buffer(player_act_pyt)
                    for b, env in enumerate(self.envs):
                        o, r, d, env_info = env.step(player_action[b])
                        if d and (env.observer_reward_shaping is not None):
                            r_obs = env.observer_reward_shaping(r,env.last_obs_act)
                        else:
                            r_obs = r
                        observer_traj_infos[b].step(observer_observation[b], observer_action[b], r_obs, d, observer_agent_info[b], env_info)
                        if getattr(env_info, "traj_done", d):
                            self.observer_traj_infos_queue.put(observer_traj_infos[b].terminate(o))
                            observer_traj_infos[b] = self.TrajInfoCls()
                            o = env.reset()
                            prev_reset[b] = True
                            if env.player_reward_shaping is not None:
                                r_ply = env.player_reward_shaping(r, env.last_obs_act)
                            else:
                                r_ply = r
                            player_traj_infos[b].step(player_observation[b], player_action[b], r_ply, d, player_agent_info[b], env_info)
                            self.player_traj_infos_queue.put(player_traj_infos[b].terminate(env.player_observation_space.null_value()))
                            player_traj_infos[b] = self.TrajInfoCls()
                        if d:
                            observer_action[b] = 0
                            player_action[b] = 0
                            r = 0
                            player_reward[b] = r
                            self.agent.reset_one(idx=b)
                        observer_observation[b] = o
                        observer_reward[b] = r
                else:
                    while not self.envs[0].player_turn:
                        pturn = self.envs[0].player_turn
                        observer_act_pyt, observer_agent_info = self.agent.step(observer_obs_pyt, observer_act_pyt, observer_rew_pyt)
                        observer_action = numpify_buffer(observer_act_pyt)
                        for b, env in enumerate(self.envs):
                            assert pturn == env.player_turn
                            o, r, d, env_info = env.step(observer_action[b])
                            assert not d
                            if env.player_turn:
                                if prev_reset[b]:
                                    prev_reset[b] = False
                                else:
                                    player_traj_infos[b].step(player_observation[b], player_action[b], r, d,
                                                                player_agent_info[b], env_info)
                                player_observation[b] = o
                                player_reward[b] = r

                            else:
                                observer_traj_infos[b].step(observer_observation[b], observer_action[b], r, d, observer_agent_info[b], env_info)
                                observer_observation[b] = o
                                observer_reward[b] = r
                if self.player_sync.stop_eval.value:
                    break

        self.observer_traj_infos_queue.put(None)  # End sentinel.
        self.player_traj_infos_queue.put(None)  # End sentinel.

