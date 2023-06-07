import os
import torch
import numpy as np
import random
import pandas as pd


def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_predictions(predictions, labels, epoch, split, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.DataFrame(data={"predictions": predictions, "labels": labels})
    pred_path = os.path.join(save_path, f"pred_split_{split}_epoch_{epoch}.csv")
    df.to_csv(pred_path)


def save_soft_embeddings(model, save_path, epoch=None):
    """Function to save soft embeddings.

    Args:
        model (nn.Module): the CSP/COOP module
        save_path (str): directory path to save the soft embeddings
        epoch (int, optional): epoch number for the soft embedding.
            Defaults to None.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the soft embedding
    with torch.no_grad():
        if epoch:
            soft_emb_path = os.path.join(save_path, f"soft_embeddings_epoch_{epoch}.pt")
        else:
            soft_emb_path = os.path.join(save_path, "soft_embeddings.pt")

        torch.save({"soft_embeddings": model.soft_embeddings}, soft_emb_path)
