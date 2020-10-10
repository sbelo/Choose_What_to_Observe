from categorical import (CategoricalPgAgent,
    RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent)
from cwto_model import CWTO_LstmModel


class CWTO_Mixin:
    """
    Mixin class defining which environment interface properties
    are given to the model.
    """

    def make_env_to_model_kwargs(self, env_spaces):
        """Extract image shape and action size."""
        return dict(observation_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class CWTO_LstmAgent(CWTO_Mixin, RecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=CWTO_LstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class CWTO_AlternatingLstmAgent(CWTO_Mixin,
        AlternatingRecurrentCategoricalPgAgent):

    def __init__(self, ModelCls=CWTO_LstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
