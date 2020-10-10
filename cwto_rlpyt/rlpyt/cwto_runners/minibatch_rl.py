
import psutil
import time
import torch
import math
from collections import deque
import numpy as np
from rlpyt.cwto_samplers.collections import BatchSpec
from rlpyt.cwto_runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
import wandb

class MinibatchRlBase(BaseRunner):
    """
    Implements startup, logging, and agent checkpointing functionality, to be
    called in the `train()` method of the subclassed runner.  Subclasses will
    modify/extend many of the methods here.

    Args:
        algo: The algorithm instance.
        agent: The learning agent instance.
        sampler: The sampler instance.
        n_steps (int): Total number of environment steps to run in training loop.
        seed (int): Random seed to use, if ``None`` will generate randomly.
        affinity (dict): Hardware component assignments for sampler and algorithm.
        log_interval_steps (int): Number of environment steps between logging to csv.
    """

    _eval = False

    def __init__(
            self,
            player_algo,
            observer_algo,
            agent,
            sampler,
            n_steps,
            seed=None,
            affinity=None,
            log_interval_steps=1e5,
            wandb_log=False,
            ):
        n_steps = int(n_steps)
        log_interval_steps = int(log_interval_steps)
        affinity = dict() if affinity is None else affinity
        save__init__args(locals())
        if self.agent.train_mask[0]:
            algo = self.player_algo
        else:
            algo = self.observer_algo
        self.min_itr_learn = getattr(algo, 'min_itr_learn', 0)

    def startup(self):
        """
        Sets hardware affinities, initializes the following: 1) sampler (which
        should initialize the agent), 2) agent device and data-parallel wrapper (if applicable),
        3) algorithm, 4) logger.
        """
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)
        if self.agent.train_mask[0]:
            algo = self.player_algo
        else:
            algo = self.observer_algo
        player_examples, observer_examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        if world_size > 1:
            self.agent.data_parallel()
        self.player_algo.initialize(
            agent=self.agent.player,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=player_examples,
            world_size=world_size,
            rank=rank,
        )
        
        obs_batch_spec = self.sampler.batch_spec
        if self.agent.serial:
            obs_batch_spec = BatchSpec(int(self.agent.n_serial*(self.sampler.batch_spec.T)),self.sampler.batch_spec.B)
        self.observer_algo.initialize(
            agent=self.agent.observer,
            n_itr=n_itr,
            batch_spec=obs_batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=observer_examples,
            world_size=world_size,
            rank=rank,
        )
        self.initialize_logging()
        return n_itr

    def get_traj_info_kwargs(self):
        """
        Pre-defines any TrajInfo attributes needed from elsewhere e.g.
        algorithm discount factor.
        """
        return dict(discount=getattr(self.player_algo, "discount", 1))

    def get_n_itr(self):
        """
        Determine number of train loop iterations to run.  Converts logging
        interval units from environment steps to iterations.
        """
        # Log at least as often as requested (round down itrs):
        log_interval_itrs = max(self.log_interval_steps //
            self.itr_batch_size, 1)
        n_itr = self.n_steps // self.itr_batch_size
        if n_itr % log_interval_itrs > 0:  # Keep going to next log itr.
            n_itr += log_interval_itrs - (n_itr % log_interval_itrs)
        self.log_interval_itrs = log_interval_itrs
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} iterations of minibatch RL.")
        return n_itr

    def initialize_logging(self):
        self._player_opt_infos = {k: list() for k in self.player_algo.opt_info_fields}
        self._observer_opt_infos = {k: list() for k in self.observer_algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._cum_time = 0.
        self._player_cum_completed_trajs = 0
        self._player_last_update_counter = 0
        self._observer_cum_completed_trajs = 0
        self._observer_last_update_counter = 0

    def shutdown(self):
        logger.log("Training complete.")
        self.pbar.stop()
        self.sampler.shutdown()

    def get_itr_snapshot(self, itr):
        """
        Returns all state needed for full checkpoint/snapshot of training run,
        including agent parameters and optimizer parameters.
        """
        return dict(
            itr=itr,
            cum_steps=itr * self.sampler.batch_size * self.world_size,
            player_agent_state_dict=self.agent.player.state_dict(),
            player_optimizer_state_dict=self.player_algo.optim_state_dict(),
            observer_agent_state_dict=self.agent.observer.state_dict(),
            observer_optimizer_state_dict=self.observer_algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def store_diagnostics(self, itr, player_traj_infos, observer_traj_infos, player_opt_info, observer_opt_info):
        """
        Store any diagnostic information from a training iteration that should
        be kept for the next logging iteration.
        """
        self._player_cum_completed_trajs += len(player_traj_infos)
        self._observer_cum_completed_trajs += len(observer_traj_infos)
        for k, v in self._player_opt_infos.items():
            new_v = getattr(player_opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        for k, v in self._observer_opt_infos.items():
            new_v = getattr(observer_opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr, player_traj_infos=None, observer_traj_infos=None, eval_time=0, prefix='Diagnostics/'):
        """
        Write diagnostics (including stored ones) to csv via the logger.
        """
        if itr > 0:
            self.pbar.stop()
        if itr >= self.min_itr_learn - 1:
            self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        train_time_elapsed = new_time - self._last_time - eval_time
        player_new_updates = self.player_algo.update_counter - self._player_last_update_counter
        observer_new_updates = self.observer_algo.update_counter - self._observer_last_update_counter
        player_new_samples = (self.sampler.batch_size * self.world_size *
            self.log_interval_itrs)
        if self.agent.serial:
            observer_new_samples = self.agent.n_serial * player_new_samples
        else:
            observer_new_samples = player_new_samples
        player_updates_per_second = (float('nan') if itr == 0 else
            player_new_updates / train_time_elapsed)
        observer_updates_per_second = (float('nan') if itr == 0 else
            observer_new_updates / train_time_elapsed)
        player_samples_per_second = (float('nan') if itr == 0 else
            player_new_samples / train_time_elapsed)
        observer_samples_per_second = (float('nan') if itr == 0 else
            observer_new_samples / train_time_elapsed)
        player_replay_ratio = (player_new_updates * self.player_algo.batch_size * self.world_size /
            player_new_samples)
        observer_replay_ratio = (observer_new_updates * self.observer_algo.batch_size * self.world_size /
            observer_new_samples)
        player_cum_replay_ratio = (self.player_algo.batch_size * self.player_algo.update_counter /
            ((itr + 1) * self.sampler.batch_size))  # world_size cancels.
        player_cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size
        if self.agent.serial:
            observer_cum_replay_ratio = (self.observer_algo.batch_size * self.observer_algo.update_counter /
                ((itr + 1) * (self.agent.n_serial * self.sampler.batch_size)))  # world_size cancels.
            observer_cum_steps = self.agent.n_serial * player_cum_steps
        else:
            observer_cum_replay_ratio = (self.observer_algo.batch_size * self.observer_algo.update_counter /
                ((itr + 1) * self.sampler.batch_size))  # world_size cancels.
            observer_cum_steps = player_cum_steps

        with logger.tabular_prefix(prefix):
            if self._eval:
                logger.record_tabular('CumTrainTime',
                    self._cum_time - self._cum_eval_time)  # Already added new eval_time.
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('CumTime (s)', self._cum_time)
            logger.record_tabular('PlayerCumSteps', player_cum_steps)
            logger.record_tabular('ObserverCumSteps', observer_cum_steps)
            logger.record_tabular('PlayerCumCompletedTrajs', self._player_cum_completed_trajs)
            logger.record_tabular('ObserverCumCompletedTrajs', self._observer_cum_completed_trajs)
            logger.record_tabular('PlayerCumUpdates', self.player_algo.update_counter)
            logger.record_tabular('ObserverCumUpdates', self.observer_algo.update_counter)
            logger.record_tabular('PlayerStepsPerSecond', player_samples_per_second)
            logger.record_tabular('ObserverStepsPerSecond', observer_samples_per_second)
            logger.record_tabular('PlayerUpdatesPerSecond', player_updates_per_second)
            logger.record_tabular('ObserverUpdatesPerSecond', observer_updates_per_second)
            logger.record_tabular('PlayerReplayRatio', player_replay_ratio)
            logger.record_tabular('ObserverReplayRatio', observer_replay_ratio)
            logger.record_tabular('PlayerCumReplayRatio', player_cum_replay_ratio)
            logger.record_tabular('ObserverCumReplayRatio', observer_cum_replay_ratio)
        self._log_infos(player_traj_infos,observer_traj_infos)
        logger.dump_tabular(with_prefix=False)

        self._last_time = new_time
        self._player_last_update_counter = self.player_algo.update_counter
        self._observer_last_update_counter = self.observer_algo.update_counter
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)

    def _log_infos(self, player_traj_infos=None, observer_traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if player_traj_infos is None:
            player_traj_infos = self._player_traj_infos
        if player_traj_infos:
            for k in player_traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat("Player" + k, [info[k] for info in player_traj_infos])

        if observer_traj_infos is None:
            observer_traj_infos = self._observer_traj_infos
        if observer_traj_infos:
            for k in observer_traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat("Observer" + k, [info[k] for info in observer_traj_infos])

        if self._player_opt_infos:
            for k, v in self._player_opt_infos.items():
                logger.record_tabular_misc_stat("Player" + k, v)
        self._player_opt_infos = {k: list() for k in self._player_opt_infos}  # (reset)

        if self._observer_opt_infos:
            for k, v in self._observer_opt_infos.items():
                logger.record_tabular_misc_stat("Observer" + k, v)
        self._observer_opt_infos = {k: list() for k in self._observer_opt_infos}  # (reset)

    def wandb_logging(self,itr,player_traj_infos=None, observer_traj_infos=None):

        if player_traj_infos is None:
            player_traj_infos = self._player_traj_infos

        if observer_traj_infos is None:
            observer_traj_infos = self._observer_traj_infos

        raw_log_dict = dict()
        if player_traj_infos:
            for k in player_traj_infos[0]:
                if not k.startswith("_"):
                    raw_log_dict["Player" + k] = np.asarray([info[k] for info in player_traj_infos])

        if observer_traj_infos:
            for k in observer_traj_infos[0]:
                if not k.startswith("_"):
                    raw_log_dict["Observer" + k] = np.asarray([info[k] for info in observer_traj_infos])

        if self._player_opt_infos:
            for k, v in self._player_opt_infos.items():
                raw_log_dict["Player" + k] = np.asarray(v)

        if self._observer_opt_infos:
            for k, v in self._observer_opt_infos.items():
                raw_log_dict["Observer" + k] = np.asarray(v)
        log_dict = dict()
        for key, value in raw_log_dict.items():
            if len(value) > 0:
                log_dict[key + "Average"] = np.average(value)
                log_dict[key + "Std"] = np.std(value)
                log_dict[key + "Median"] = np.median(value)
                log_dict[key + "Min"] = np.min(value)
                log_dict[key + "Max"] = np.max(value)
            else:
                log_dict[key + "Average"] = np.nan
                log_dict[key + "Std"] = np.nan
                log_dict[key + "Median"] = np.nan
                log_dict[key + "Min"] = np.nan
                log_dict[key + "Max"] = np.nan
        wandb.log(log_dict,step=itr)

class MinibatchRl(MinibatchRlBase):
    """
    Runs RL on minibatches; tracks performance online using learning
    trajectories.
    """

    def __init__(self, log_traj_window=100, **kwargs):
        """
        Args: 
            log_traj_window (int): How many trajectories to hold in deque for computing performance statistics.
        """
        super().__init__(**kwargs)
        self.log_traj_window = int(log_traj_window)

    def train(self):
        """
        Performs startup, then loops by alternating between
        ``sampler.obtain_samples()`` and ``algo.optimize_agent()``, logging
        diagnostics at the specified interval.
        """
        n_itr = self.startup()
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                player_samples, player_traj_infos, observer_samples, observer_traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                player_opt_info = ()
                observer_opt_info = ()
                if self.agent.train_mask[0]:
                    player_opt_info = self.player_algo.optimize_agent(itr, player_samples)
                if self.agent.train_mask[1]:
                    observer_opt_info = self.observer_algo.optimize_agent(itr, observer_samples)
                self.store_diagnostics(itr, player_traj_infos, observer_traj_infos, player_opt_info, observer_opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    if self.wandb_log:
                        self.wandb_logging(itr)
                    self.log_diagnostics(itr)
        self.shutdown()

    def initialize_logging(self):
        self._player_traj_infos = deque(maxlen=self.log_traj_window)
        self._observer_traj_infos = deque(maxlen=self.log_traj_window)
        self._player_new_completed_trajs = 0
        self._observer_new_completed_trajs = 0
        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, player_traj_infos, observer_traj_infos, player_opt_info, observer_opt_info):
        self._player_new_completed_trajs += len(player_traj_infos)
        self._observer_new_completed_trajs += len(observer_traj_infos)
        self._player_traj_infos.extend(player_traj_infos)
        self._observer_traj_infos.extend(observer_traj_infos)
        super().store_diagnostics(itr, player_traj_infos, observer_traj_infos, player_opt_info, observer_opt_info)

    def log_diagnostics(self, itr, prefix='Diagnostics/'):
        with logger.tabular_prefix(prefix):
            logger.record_tabular('PlayerNewCompletedTrajs', self._player_new_completed_trajs)
            logger.record_tabular('ObserverNewCompletedTrajs', self._observer_new_completed_trajs)
            logger.record_tabular('PlayerStepsInTrajWindow',
                sum(info["Length"] for info in self._player_traj_infos))
            logger.record_tabular('ObserverStepsInTrajWindow',
                sum(info["Length"] for info in self._observer_traj_infos))
        super().log_diagnostics(itr, prefix=prefix)
        self._player_new_completed_trajs = 0
        self._observer_new_completed_trajs = 0


class MinibatchRlEval(MinibatchRlBase):
    """
    Runs RL on minibatches; tracks performance offline using evaluation
    trajectories.
    """

    _eval = True

    def train(self):
        """
        Performs startup, evaluates the initial agent, then loops by
        alternating between ``sampler.obtain_samples()`` and
        ``algo.optimize_agent()``.  Pauses to evaluate the agent at the
        specified log interval.
        """
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            player_eval_traj_infos,observer_eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, player_eval_traj_infos, observer_eval_traj_infos, eval_time)
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                player_samples, player_traj_infos, observer_samples, observer_traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                player_opt_info = ()
                observer_opt_info = ()
                if self.agent.train_mask[0]:
                    player_opt_info = self.player_algo.optimize_agent(itr, player_samples)
                if self.agent.train_mask[1]:
                    observer_opt_info = self.observer_algo.optimize_agent(itr, observer_samples)
                self.store_diagnostics(itr, player_traj_infos, observer_traj_infos, player_opt_info, observer_opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    player_eval_traj_infos, observer_eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    if self.wandb_log:
                        self.wandb_logging(itr,player_traj_infos=player_eval_traj_infos,observer_traj_infos=observer_eval_traj_infos)
                    self.log_diagnostics(itr, player_eval_traj_infos, observer_eval_traj_infos, eval_time)

        self.shutdown()

    def evaluate_agent(self, itr):
        """
        Record offline evaluation of agent performance, by ``sampler.evaluate_agent()``.
        """
        if itr > 0:
            self.pbar.stop()

        if itr >= self.min_itr_learn - 1 or itr == 0:
            logger.log("Evaluating agent...")
            self.agent.eval_mode(itr)  # Might be agent in sampler.
            eval_time = -time.time()
            player_traj_infos, observer_traj_infos = self.sampler.evaluate_agent(itr)
            eval_time += time.time()
        else:
            player_traj_infos = []
            observer_traj_infos = []
            eval_time = 0.0
        logger.log("Evaluation runs complete.")
        return player_traj_infos, observer_traj_infos, eval_time

    def initialize_logging(self):
        super().initialize_logging()
        self._cum_eval_time = 0

    def log_diagnostics(self, itr, player_eval_traj_infos, observer_eval_traj_infos, eval_time, prefix='Diagnostics/'):
        if not player_eval_traj_infos:
            logger.log("WARNING: player had no complete trajectories in eval.")
        if not observer_eval_traj_infos:
            logger.log("WARNING: observer had no complete trajectories in eval.")
        player_steps_in_eval = sum([info["Length"] for info in player_eval_traj_infos])
        observer_steps_in_eval = sum([info["Length"] for info in observer_eval_traj_infos])
        with logger.tabular_prefix(prefix):
            logger.record_tabular('PlayerStepsInEval', player_steps_in_eval)
            logger.record_tabular('ObserverStepsInEval', observer_steps_in_eval)
            logger.record_tabular('PlayerTrajsInEval', len(player_eval_traj_infos))
            logger.record_tabular('ObserverTrajsInEval', len(observer_eval_traj_infos))
            self._cum_eval_time += eval_time
            logger.record_tabular('CumEvalTime', self._cum_eval_time)
        super().log_diagnostics(itr, player_eval_traj_infos, observer_eval_traj_infos, eval_time, prefix=prefix)
