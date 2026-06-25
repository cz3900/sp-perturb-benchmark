"""Celcomen as an offline end-to-end loader. Celcomen runs in its own python=3.9 env (CCE training +
Simcomen counterfactual generation) and dumps a per-guide `.h5ad` whose counterfactual expression
lives in layers['counterfactual']. Loading + bystander-niche extraction is identical to SpatialProp's
so we subclass it and only change the model name + default layer."""
from .spatialprop_model import SpatialPropModel


class CelcomenModel(SpatialPropModel):
    name = "celcomen"

    def __init__(self, prediction_paths, layer="counterfactual"):
        super().__init__(prediction_paths, layer)
