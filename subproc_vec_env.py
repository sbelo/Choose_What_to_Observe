import multiprocessing
from collections import OrderedDict
from typing import Sequence
from env_wrapper import *

import gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                stat = env.step(data)
                if isinstance(stat,CHECK_):
                    remote.send(stat.data)
                else:
                    stat = list(stat)
                    if stat[2]:
                        # save final observation where user can get it, then reset
                        stat[3]["terminal_observation"] = stat[0]
                        stat[0] = env.reset()
                    remote.send(tuple(stat))
            elif cmd == "player_step":
                observation, reward, done, info = env.player_step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "update_budget":
                env.update_budget(data)
            elif cmd == "is_ser_done":
                remote.send(env.ser_done)
            elif cmd == "get_last_action":
                remote.send(env.last_action)
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "set_last_obs":
                env.set_last_obs(data)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, player_flag, model = None, lstm_flag = True, start_method=None, budget_constraint=None, serial=False):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.model = model
        self.num_envs = n_envs
        self.serial = serial
        self.player_flag = player_flag
        self.lstm_flag = lstm_flag
        self.budget_constraint = budget_constraint
        if not player_flag:
            self._last_dones = None
        if lstm_flag:
            self.last_hiddens = None

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in multiprocessing.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def set_model(self,model):
        self.model = model
    def player_step_async(self,actions):
        if actions is None:
            for remote in self.remotes:
                remote.send(("player_step", None))
        else:
            for remote, action in zip(self.remotes, actions):
                remote.send(("player_step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        if self.player_flag:
            obs, rews, dones, infos = zip(*results)
            obs = _flatten_obs(obs, self.observation_space)
            if self.serial:
                ser_obs = np.concatenate([np.zeros(obs.shape),self._last_masked],axis=1)
                for dd in range(obs.shape[1]):
                    ser_obs[:,dd] = self.model.i_predict(ser_obs,dones)
                masks = ser_obs[:,:obs.shape[1]]
            else:
                masks = self.model.i_predict(np.concatenate([self._last_masks,self._last_masked],axis=1),dones)
            self._last_masks = masks
            self._last_masked = np.multiply(masks,obs)
            obs = np.concatenate([masks,self._last_masked],axis=1)

            for remote, act in zip(self.remotes,masks):
                remote.send(("update_budget",act))

        else:
            obs = _flatten_obs(results, self.observation_space)
            valid_predict = True
            if self.serial:
                for remote in self.remotes:
                    remote.send(("is_ser_done", None))
                ser_dones = np.asarray([remote.recv() for remote in self.remotes])
                if False in ser_dones:
                    valid_predict = False
            if valid_predict:
                for remote in self.remotes:
                    remote.send(("get_last_action", None))
                last_actions = np.stack([remote.recv() for remote in self.remotes])
                actions = self.model.i_predict(np.concatenate([last_actions,obs[:,int(obs.shape[1]/2):]],axis=1),self._last_dones)
                self.player_step_async(actions)
                results2 = [remote.recv() for remote in self.remotes]
                last_obs, rews, dones, infos = zip(*results2)
                self._last_dones = dones
                if self.budget_constraint is not None:
                    cont_rews = []
                    for ee in range(self.num_envs):
                        cont_rews.append(self.budget_constraint(rews[ee], last_actions[ee], self.model.prices))
                    rews = np.stack(cont_rews)
            else:
                rews = np.zeros(self.num_envs)
                if self._last_dones is None:
                    dones = [False for _ in range(self.num_envs)]
                else:
                    dones = self._last_dones
                infos = [{} for _ in range(self.num_envs)]

        self.waiting = False
        return obs, np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_obs(obs, self.observation_space)
        self._last_masked = 0 * obs
        if self.player_flag:
            self._last_masks = 0 * obs
            obs = np.concatenate([obs,obs],axis=1)
        return obs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)
