import clip
import torch
from .add import AddModel, AddRelModel
from .mult import MultModel, MultRelModel
from .conv import SimpleConvModel, SimpleConvRelModel
from .rf import RFConvModel, RFConvRelModel
from .tl import MatrixModel, MatrixRelObjModel

REL_MODELS = {
    "add": AddRelModel,
    "mult": MultRelModel,
    "conv": SimpleConvRelModel,
    "rf": RFConvRelModel,
    "tl": MatrixRelObjModel,
}

OBJ_MODELS = {
    "add": AddModel,
    "mult": MultModel,
    "conv": SimpleConvModel,
    "rf": RFConvModel,
    "tl": MatrixModel,
}


def relational_cdsm(clip_model, train_dataset, config, device):
    reltoi = {rel: i for i, rel in enumerate(train_dataset.relations)}
    nountoi = {noun: i for i, noun in enumerate(train_dataset.nouns)}

    model = REL_MODELS[config.model_name](clip_model, reltoi, nountoi, config, device)

    return model


def object_cdsm(clip_model, train_dataset, config, device):
    adjtoi = {adj: i for i, adj in enumerate(train_dataset.attributes)}
    nountoi = {noun: i for i, noun in enumerate(train_dataset.nouns)}

    model = OBJ_MODELS[config.model_name](clip_model, adjtoi, nountoi, config, device)

    return model


def get_cdsm(train_dataset, config, device):
    # instantiate CLIP model
    clip_model, preprocess = clip.load(config.clip_model, device=device)

    # freeze all the parameters of the CLIP model
    for p in clip_model.parameters():
        p.requires_grad = False

    if config.dataset == "rel":
        model = relational_cdsm(clip_model, train_dataset, config, device)
    else:
        model = object_cdsm(clip_model, train_dataset, config, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    return model, optimizer
