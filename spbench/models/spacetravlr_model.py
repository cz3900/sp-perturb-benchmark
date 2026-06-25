"""SpaceTravLR as an offline end-to-end loader. SpaceTravLR runs in its own uv env (celloracle/commot)
and dumps a per-guide `.h5ad` whose PANEL-ALIGNED perturbed prediction lives in
layers['predicted_perturbed']. The loading + bystander-niche extraction is identical to SpatialProp's,
so we subclass it and only change the model name + default layer."""
from .spatialprop_model import SpatialPropModel


class SpaceTravLRModel(SpatialPropModel):
    name = "spacetravlr"

    def __init__(self, prediction_paths, layer="predicted_perturbed"):
        super().__init__(prediction_paths, layer)
