from stable_baselines import PPO2
from stable_baselines.common.policies import *

def raw_predict1(model, obs, dones, rnnhs):
    actions, _, new_rnnhs, _ = model.step(obs, state=rnnhs, mask=dones)
    return actions, new_rnnhs


class GENERAL_WRAPPER(PPO2):
    def __init__(self, raw_predict, **kwargs):
        super().__init__(**kwargs)
        self.raw_predict = raw_predict
        self.last_rnnh = self.initial_state
        self._random = False
        self._fixed_flag = False

    def i_predict(self,obs,dones):
        if self._fixed_flag:
            return self.fixed_action
        elif self._random:
            pred = [self.env.action_space.sample() for _ in range(self.n_envs)]
            pred = np.stack(pred)
        else:
            if dones is None:
                dones = np.zeros(len(obs),dtype=bool)
            pred, new_rnnh = self.raw_predict(self,obs,dones,self.last_rnnh)
            self.last_rnnh = new_rnnh
        return pred

    def random_on(self):
        self._random = True

    def random_off(self):
        self._random = False

    def fixed_on(self,action):
        self.fixed_action = np.stack([action for _ in range(self.n_envs)])
        self._fixed_flag = True

    def fixed_off(self):
        self._fixed_flag = False
## gets model, observation as numpy array and optionally (if recurrent) rnnh as the memory of the recurrent layers.
## returns the prediction as numpy array and optionally the new memory of the recurrent layers
    # def raw_predict(model,obs,rnnh):

        # return pred, new_rnnh


