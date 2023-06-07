import os
import torch

from .clip import get_clip_model
from .csp import get_csp
from .cdsm import get_cdsm


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device):
    if config.model_name == "clip":
        return get_clip_model(config, device)
    elif config.model_name == "csp":
        return get_csp(train_dataset, config, device)
    elif (
        config.model_name == "add"
        or config.model_name == "mult"
        or config.model_name == "conv"
        or config.model_name == "rf"
        or config.model_name == "tl"
    ):
        return get_cdsm(train_dataset, config, device)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(config.model_name)
        )
