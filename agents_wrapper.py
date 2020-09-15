from stable_baselines import PPO2
import numpy as np
from stable_baselines.common.policies import *

def raw_predict1(model, obs, dones, rnnhs):
    actions, _, new_rnnhs, _ = model.step(obs, state=rnnhs, mask=dones)
    return actions, new_rnnhs


class GENERAL_WRAPPER(PPO2):
    def __init__(self, raw_predict, **kwargs):
        super().__init__(**kwargs)
        self.raw_predict = raw_predict
        self.last_rnnh = self.initial_state

    def i_predict(self,obs,dones):
        if dones is None:
            dones = np.zeros(len(obs),dtype=bool)
        pred, new_rnnh = self.raw_predict(self,obs,dones,self.last_rnnh)
        self.last_rnnh = new_rnnh
        return pred

## gets model, observation as numpy array and optionally (if recurrent) rnnh as the memory of the recurrent layers.
## returns the prediction as numpy array and optionally the new memory of the recurrent layers
    # def raw_predict(model,obs,rnnh):

        # return pred, new_rnnh


