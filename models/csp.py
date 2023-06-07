# Code obtained from https://github.com/BatsResearch/csp

import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from clip.model import convert_weights
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class CSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings,
        class_token_ids,
        concept_to_idx,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)
        self.concept_to_idx = concept_to_idx

    def construct_token_tensors(self, texts):
        """Function creates the token tensor for further inference.
        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj
        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        pairs = [pair.split() for pairs in texts for pair in pairs]
        pairs = [(self.concept_to_idx[a], self.concept_to_idx[n]) for a, n in pairs]
        pair_idx = torch.tensor(pairs, dtype=torch.long, device=self.device)

        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[attr_idx].type(
            self.clip_model.dtype
        )
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[obj_idx + self.offset].type(
            self.clip_model.dtype
        )

        return token_tensor

    def compute_text_representations(self, texts):
        token_tensors = self.construct_token_tensors(texts)

        text_features = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )
        return text_features

    def forward(self, batch_images, texts):
        texts = list(map(list, zip(*texts)))
        bsz = len(texts)
        num_captions = len(texts[0])

        text_features = self.compute_text_representations(texts)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view([bsz, num_captions, -1])

        batch_images = batch_images.to(self.device)
        batch_feat = self.encode_image(batch_images)

        batch_feat = batch_feat / batch_feat.norm(dim=-1, keepdim=True)
        batch_feat = batch_feat.unsqueeze(2)

        logits_per_image = self.clip_model.logit_scale.exp() * torch.bmm(
            text_features, batch_feat
        )
        logits_per_image = logits_per_image.squeeze(2)
        return logits_per_image


class RelCSPInterface(CSPInterface):
    def construct_token_tensors(self, texts):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 3], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        pairs = [pair.split() for pairs in texts for pair in pairs]
        pairs = [
            (self.concept_to_idx[a], self.concept_to_idx[r], self.concept_to_idx[b])
            for a, r, b in pairs
        ]
        pair_idx = torch.tensor(pairs, dtype=torch.long, device=self.device)

        a_idx, rel_idx, b_idx = pair_idx[:, 0], pair_idx[:, 1], pair_idx[:, 2]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())

        soft_embeddings = self.attr_dropout(self.soft_embeddings)

        token_tensor[:, eos_idx - 3, :] = soft_embeddings[a_idx].type(
            self.clip_model.dtype
        )
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[rel_idx + self.offset].type(
            self.clip_model.dtype
        )
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[b_idx].type(
            self.clip_model.dtype
        )

        return token_tensor


def csp_init(
    train_dataset,
    config,
    device,
    prompt_template="a photo of X X X",
):
    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    concepts = train_dataset.concepts

    tokenized = torch.cat(
        [clip.tokenize(tok, context_length=config.context_length) for tok in concepts]
    )

    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

    soft_embedding = torch.zeros(
        (len(concepts), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    soft_embedding = nn.Parameter(soft_embedding)

    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )
    # offset = len(objects)
    offset = 0

    return (clip_model, soft_embedding, class_token_ids, offset)


def object_model(train_dataset, config, device):
    prompt_template = "a photo of X X"
    (clip_model, soft_embedding, class_token_ids, offset) = csp_init(
        train_dataset, config, device, prompt_template=prompt_template
    )
    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        train_dataset.concept_to_idx,
        device,
        attr_dropout=config.attr_dropout,
    )
    optimizer = torch.optim.Adam(
        [soft_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    return interface, optimizer


def relational_model(train_dataset, config, device):
    prompt_template = "a photo of X X X"
    (clip_model, soft_embedding, class_token_ids, offset) = csp_init(
        train_dataset, config, device, prompt_template=prompt_template
    )
    interface = RelCSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        train_dataset.concept_to_idx,
        device,
        attr_dropout=config.attr_dropout,
    )
    optimizer = torch.optim.Adam(
        [soft_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    return interface, optimizer


def get_csp(train_dataset, config, device):
    if config.dataset in [
        "single-object",
        "two-object",
    ]:
        interface, optimizer = object_model(train_dataset, config, device)
    else:
        interface, optimizer = relational_model(train_dataset, config, device)

    return interface, optimizer
