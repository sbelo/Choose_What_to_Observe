import numpy as np
import wandb
import torch



class CWTO_AgentWrapper:
    def __init__(self, player, observer, serial = False, n_serial = None, alt = False, train_mask=[True,True]):
        self._player_turn = False
        self.player = player
        self.player.is_player = True
        self.observer = observer
        self.observer.is_player = False
        self.alt = alt
        self.train_mask = train_mask
        if alt:
            self.alt_counter = 2
        self.serial = serial
        self.n_serial = n_serial
        if serial:
            assert n_serial is not None
            self.curr_ser_count = n_serial

    def reset(self):
        if self.alt:
            self.alt_counter = 2
        if self.serial:
            self.curr_ser_count = self.n_serial
        self._player_turn = False
        self.player.reset()
        self.observer.reset()

    def reset_one(self,idx):
        self.player.reset_one(idx)
        self.observer.reset_one(idx)

    def eval_mode(self,itr):
        self.player.eval_mode(itr)
        self.observer.eval_mode(itr)

    def sample_mode(self,itr):
        self.player.sample_mode(itr)
        self.observer.sample_mode(itr)

    def collector_initialize(self,**kwargs):
        self.player.collector_initialize(**kwargs)
        self.observer.collector_initialize(**kwargs)

    def value(self, obs_pyt, act_pyt, rew_pyt, is_player):
        if is_player:
            return self.player.value(obs_pyt,act_pyt,rew_pyt)
        else:
            return self.observer.value(obs_pyt, act_pyt, rew_pyt)

    def train_mode(self,itr):
        if self.train_mask[0]:
            self.player.train_mode(itr)
        else:
            self.player.sample_mode(itr)
        if self.train_mask[1]:
            self.observer.train_mode(itr)
        else:
            self.observer.sample_mode(itr)

    def data_parallel(self):
        self.player.data_parallel()
        self.observer.data_parallel()

    def step(self,player_obs_pyt, player_act_pyt, player_rew_pyt,player_turn=None):
        change_flag = False
        if player_turn is None:
            change_flag = True
            player_turn = self._player_turn

        if player_turn:
            res = self.player.step(player_obs_pyt, player_act_pyt, player_rew_pyt)
            if self.alt:
                next_p_turn = True
                self.alt_counter -= 1
                if self.alt_counter <= 0:
                    self.alt_counter = 2
                    next_p_turn = False
            else:
                next_p_turn = False
        else:
            res = self.observer.step(player_obs_pyt, player_act_pyt, player_rew_pyt)
            if self.serial:
                next_p_turn = False
                if self.alt:
                    self.alt_counter -= 1
                    if self.alt_counter <= 0:
                        self.alt_counter = 2
                        self.curr_ser_count -= 1
                else:
                    self.curr_ser_count -= 1
                if self.curr_ser_count <= 0:
                    next_p_turn = True
                    self.curr_ser_count = self.n_serial
            else:
                if self.alt:
                    next_p_turn = False
                    self.alt_counter -= 1
                    if self.alt_counter <= 0:
                        self.alt_counter = 2
                        next_p_turn = True
                else:
                    next_p_turn = True
        if change_flag:
            self._player_turn = next_p_turn
        return res

    def toggle_alt(self):
        self.player.toggle_alt()
        self.observer.toggle_alt()

    def to_device(self,cuda_idx=None):
        if isinstance(cuda_idx,list):
            self.player.to_device(cuda_idx=cuda_idx[0])
            self.observer.to_device(cuda_idx=cuda_idx[1])
        else:
            self.player.to_device(cuda_idx=cuda_idx)
            self.observer.to_device(cuda_idx=cuda_idx)

    def async_cpu(self, share_memory=True):
        self.player.async_cpu(share_memory=share_memory)
        self.observer.async_cpu(share_memory=share_memory)

    def sync_shared_memory(self):
        self.player.sync_shared_memory()
        self.observer.sync_shared_memory()
    def send_shared_memory(self):
        self.player.send_shared_memory()
        self.observer.send_shared_memory()
    def recv_shared_memory(self):
        self.player.recv_shared_memory()
        self.observer.recv_shared_memory()
    def player_turn(self):
        return self._player_turn
    def save_models_to_wandb(self):
        torch.save(self.player.model.state_dict(), "player_model")
        wandb.save('player_model')
        torch.save(self.observer.model.state_dict(), "observer_model")
        wandb.save('observer_model')


