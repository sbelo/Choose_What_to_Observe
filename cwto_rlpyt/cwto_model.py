import torch
import torch.nn.functional as F
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.models.mlp import MlpModel
import numpy as np

RnnState = namedarraytuple("RnnState", ["h", "c"])

class CWTO_LstmModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            output_size,
            hidden_sizes=None,  # None for default (see below).
            lstm_size=256,
            nonlinearity=torch.nn.ReLU,
            normalize_observation=False,
            norm_obs_clip=10,
            norm_obs_var_clip=1e-6,
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self._obs_n_dim = len(observation_shape)
        hidden_sizes = hidden_sizes or [256, 256]
        mlp_input_size = int(np.prod(observation_shape))
        self.mlp = MlpModel(
            input_size=mlp_input_size,
            hidden_sizes=hidden_sizes,
            output_size=None,
            nonlinearity=nonlinearity,
        )

        mlp_output_size = hidden_sizes[-1] if hidden_sizes else mlp_input_size
        self.lstm = torch.nn.LSTM(mlp_output_size + output_size + 1, lstm_size)
        self.pi = torch.nn.Linear(lstm_size, output_size)
        self.value = torch.nn.Linear(lstm_size, 1)
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.norm_obs_clip = norm_obs_clip
            self.norm_obs_var_clip = norm_obs_var_clip
        self.normalize_observation = normalize_observation

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_n_dim)

        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                                      obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        mlp_out = self.mlp(observation.view(T * B, -1))
        lstm_input = torch.cat([
            mlp_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
        v = self.value(lstm_out.view(T * B, -1)).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return pi, v, next_rnn_state