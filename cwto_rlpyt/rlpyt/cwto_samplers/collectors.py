
import numpy as np

from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args


class BaseCollector:
    """Class that steps environments, possibly in worker process."""

    def __init__(
            self,
            rank,
            envs,
            player_samples_np,
            observer_samples_np,
            batch_T,
            TrajInfoCls,
            agent=None,  # Present or not, depending on collector class.
            player_sync=None,
            observer_sync=None,
            player_step_buffer_np=None,
            observer_step_buffer_np=None,
            global_B=1,
            env_ranks=None,
            ):
        save__init__args(locals())

    def start_envs(self):
        """e.g. calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        """In CPU-collectors, call ``agent.collector_initialize()`` e.g. to set up
        vector epsilon-greedy, and reset the agent.
        """
        if getattr(self, "agent", None) is not None:  # Not in GPU collectors.
            self.agent.collector_initialize(
                global_B=self.global_B,  # Args used e.g. for vector epsilon greedy.
                env_ranks=self.env_ranks,
            )
            self.agent.reset()
            self.agent.sample_mode(itr=0)

    def collect_batch(self, player_agent_inputs, observer_agent_inputs, player_traj_infos, observer_traj_infos):
        """Main data collection loop."""
        raise NotImplementedError

    def reset_if_needed(self, player_agent_inputs, observer_agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        pass


class BaseEvalCollector:
    """Collectors for offline agent evalution; not to record intermediate samples."""

    def __init__(
            self,
            rank,
            envs,
            TrajInfoCls,
            player_traj_infos_queue,
            observer_traj_infos_queue,
            max_T,
            agent=None,
            player_sync=None,
            observer_sync=None,
            player_step_buffer_np=None,
            observer_step_buffer_np=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self):
        """Run agent evaluation in environment and return completed trajectory
        infos."""
        raise NotImplementedError


class DecorrelatingStartCollector(BaseCollector):
    """Collector which can step all environments through a random number of random
    actions during startup, to decorrelate the states in training batches.
    """

    def start_envs(self, max_decorrelation_steps=0):
        """Calls ``reset()`` on every environment instance, then steps each
        one through a random number of random actions, and returns the
        resulting agent_inputs buffer (`observation`, `prev_action`,
        `prev_reward`)."""
        player_traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observer_traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        player_observations = list()
        observer_observations = list()
        for env in self.envs:
            observer_observations.append(env.reset())
            player_observations.append(env.player_observation_space.null_value())
        observer_observation = buffer_from_example(observer_observations[0], len(self.envs))
        player_observation = buffer_from_example(player_observations[0], len(self.envs))
        for b, obs in enumerate(observer_observations):
            observer_observation[b] = obs  # numpy array or namedarraytuple
        player_prev_action = np.stack([env.player_action_space.null_value()
            for env in self.envs])
        observer_prev_action = np.stack([env.observer_action_space.null_value()
                                       for env in self.envs])
        player_prev_reward = np.zeros(len(self.envs), dtype="float32")
        observer_prev_reward = np.zeros(len(self.envs), dtype="float32")
        player_done = np.zeros(len(self.envs), dtype=bool)
        observer_done = np.zeros(len(self.envs), dtype=bool)
        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}")
        if max_decorrelation_steps != 0:

            for b, env in enumerate(self.envs):
                n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
                if n_steps % 2 != 0:
                    if n_steps < max_decorrelation_steps or n_steps <= 1:
                        n_steps += 1
                    else:
                        n_steps -= 1
                for cstep in range(n_steps):
                    if env.player_turn:

                        a = env.action_space().sample()
                        o, r, d, info = env.step(a)
                        player_prev_action[b] = a
                        observer_prev_reward[b] = r
                        observer_done[b] = d
                        if cstep > 0:
                            observer_traj_infos[b].step(observer_observation[b], observer_prev_action[b], observer_prev_reward[b], observer_done[b], None, info)
                        if d:
                            o = env.reset()
                            observer_prev_reward[b] = 0
                            observer_traj_infos[b] = self.TrajInfoCls()
                            player_prev_reward[b] = 0
                            player_traj_infos[b] = self.TrajInfoCls()
                            player_done[b] = d
                        observer_observation[b] = o
                    else:
                        if env.serial:
                            while not env.player_turn:
                                a = env.action_space().sample()
                                o, r, d, info = env.step(a)
                                assert not d
                                observer_prev_action[b] = a
                                if env.player_turn:
                                    player_prev_reward[b] = r
                                    
                                    player_done[b] = d
                                    if cstep > 0:
                                        player_traj_infos[b].step(player_observation[b], player_prev_action[b],
                                                                  player_prev_reward[b], player_done[b], None, info)
                                    player_observation[b] = o
                                else:
                                    observer_prev_reward[b] = r
                                    observer_done[b] = d
                                    if cstep > 0:
                                        observer_traj_infos[b].step(observer_observation[b], observer_prev_action[b],
                                                                  observer_prev_reward[b], observer_done[b], None, info)
                                    observer_observation[b] = o

                        else:
                            a = env.action_space().sample()
                            o, r, d, info = env.step(a)
                            assert not d
                            observer_prev_action[b] = a
                            player_prev_reward[b] = r
                            player_done[b] = d
                            if cstep > 0:
                                player_traj_infos[b].step(player_observation[b], player_prev_action[b], player_prev_reward[b], player_done[b], None, info)
                            
                            player_observation[b] = o

        # For action-server samplers.
        if hasattr(self, "observer_step_buffer_np") and self.observer_step_buffer_np is not None:
            self.observer_step_buffer_np.observation[:] = observer_observation
            self.observer_step_buffer_np.action[:] = observer_prev_action
            self.observer_step_buffer_np.reward[:] = observer_prev_reward
        if hasattr(self, "player_step_buffer_np") and self.player_step_buffer_np is not None:
            self.player_step_buffer_np.observation[:] = player_observation
            self.player_step_buffer_np.action[:] = player_prev_action
            self.player_step_buffer_np.reward[:] = player_prev_reward

        return AgentInputs(player_observation, player_prev_action, player_prev_reward), player_traj_infos, AgentInputs(observer_observation, observer_prev_action, observer_prev_reward), observer_traj_infos
