
import numpy as np

from rlpyt.cwto_samplers.collectors import BaseEvalCollector
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args

# For sampling, serial sampler can use Cpu collectors.


class SerialEvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            max_T,
            max_trajectories=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        player_traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observer_traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        player_completed_traj_infos = list()
        observer_completed_traj_infos = list()
        observer_observations = list()
        player_observations = list()
        for env in self.envs:
            observer_observations.append(env.reset())
            player_observations.append(env.player_observation_space.null_value())
        observer_observation = buffer_from_example(observer_observations[0], len(self.envs))
        player_observation = buffer_from_example(player_observations[0], len(self.envs))
        for b, o in enumerate(observer_observations):
            observer_observation[b] = o
        observer_action = buffer_from_example(self.envs[0].observer_action_space.null_value(),len(self.envs))
        player_action = buffer_from_example(self.envs[0].player_action_space.null_value(),len(self.envs))
        observer_reward = np.zeros(len(self.envs), dtype="float32")
        player_reward = np.zeros(len(self.envs), dtype="float32")

        observer_obs_pyt, observer_act_pyt, observer_rew_pyt = torchify_buffer((observer_observation, observer_action, observer_reward))
        player_obs_pyt, player_act_pyt, player_rew_pyt = torchify_buffer((player_observation, player_action, player_reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        prev_reset = np.ones(len(self.envs), dtype=bool)
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

                        observer_traj_infos[b].step(observer_observation[b], observer_action[b], r_obs, d,
                                           observer_agent_info[b], env_info)
                        if getattr(env_info, "traj_done", d):
                            observer_completed_traj_infos.append(observer_traj_infos[b].terminate(o))
                            observer_traj_infos[b] = self.TrajInfoCls()

                            if env.player_reward_shaping is not None:
                                r_ply = env.player_reward_shaping(r, env.last_obs_act)
                            else:
                                r_ply = r
                            player_traj_infos[b].step(player_observation[b], player_action[b], r_ply, d, player_agent_info[b], env_info)
                            player_completed_traj_infos.append(player_traj_infos[b].terminate(env.player_observation_space.null_value()))
                            player_traj_infos[b] = self.TrajInfoCls()
                            prev_reset[b] = True
                            o = env.reset()
                        if d:
                            observer_action[b] = 0  # Prev_action for next step.
                            player_action[b] = 0  # Prev_action for next step.
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
                                observer_traj_infos[b].step(observer_observation[b], observer_action[b], r, d,
                                                                        observer_agent_info[b], env_info)
                                observer_observation[b] = o
                                observer_reward[b] = r


                if (self.max_trajectories is not None and
                        len(player_completed_traj_infos) >= self.max_trajectories):
                    logger.log("Evaluation reached max num trajectories "
                        f"({self.max_trajectories}).")
                    break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                f"({self.max_T}).")
        return player_completed_traj_infos, observer_completed_traj_infos
