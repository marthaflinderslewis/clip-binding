import json
import random
import pandas as pd
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .read_datasets import DATASET_PATHS

BICUBIC = InterpolationMode.BICUBIC
n_px = 224

preprocess = Compose(
    [
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

REL_PARAMS = {
    "relations": ["front", "behind", "left", "right"],
    "rel-opposites": {
        "front": "behind",
        "behind": "front",
        "left": "right",
        "right": "left",
    },
    "nouns": ["cube", "sphere", "cylinder"],
}


class ObjectDataset(Dataset):
    def __init__(self, split, dataset):
        self.split = split
        self.dataset = dataset
        # img_dir is the directory where the actual images are stored
        self.img_dir = DATASET_PATHS[dataset][f"{split}_image_path"]

        # df is a pandas dataframe that contains the labels for each image
        self.df = pd.read_csv(DATASET_PATHS[dataset][f"{self.split}_label_path"])

        self.attributes = [
            "blue",
            "brown",
            "cyan",
            "gray",
            "green",
            "purple",
            "red",
            "yellow",
        ]
        self.nouns = ["cube", "sphere", "cylinder"]

        self.concepts = self.attributes + self.nouns
        self.concept_to_idx = dict(
            [(concept, i) for i, concept in enumerate(self.concepts)]
        )

    def __len__(self):
        # the length of the dataset is the total number of positive labels (differs per image)
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.df.iloc[idx]["file_name"]
        texts = self.df.iloc[idx][["pos", "neg_0", "neg_1", "neg_2", "neg_3"]].tolist()

        image = Image.open(img_path)  # Image from PIL module
        # transform image to tensor so we can return it
        image = preprocess(image)
        label = 0

        # returns image tensor, possible captions, label of correct caption
        return image, texts, label


class RelDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.img_dir = DATASET_PATHS["rel"][f"{split}_image_path"]

        # load the labels from the json file
        label_file = DATASET_PATHS["rel"][f"{split}_label_path"]
        with open(label_file, "r") as l:
            self.labels = json.load(l)

        self.ims_labels = [
            (im, p) for im in self.labels for p in self.labels[im]["pos"]
        ]

        self.rel_opposites = REL_PARAMS["rel-opposites"]
        self.nouns = REL_PARAMS["nouns"]
        self.objects = REL_PARAMS["nouns"]
        self.relations = REL_PARAMS["relations"]

        self.concepts = self.objects + self.relations
        self.concept_to_idx = dict(
            [(concept, i) for i, concept in enumerate(self.concepts)]
        )

    def __len__(self):
        # the length of the dataset is the total number of positive labels
        return len(self.ims_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir + self.ims_labels[idx][0]
        image = Image.open(img_path)  # Image from PIL module
        image = preprocess(image)

        # for regular train/validation/testing
        subj, rel, obj = self.ims_labels[idx][1].strip().split()

        # Distractors have the following structure. If the true label is aRb,
        # the distractors are bRa, aSb, cRb, aRc, where S is the opposite relation to R
        # and c is an object not equal to a or b
        distractors = []
        distractors.append(f"{obj} {rel} {subj}")
        distractors.append(f"{subj} {self.rel_opposites[rel]} {obj}")
        # since there are always three nouns, this shouldn't make a difference.
        other_nouns = list(set(self.nouns).difference(set([subj, obj])))
        assert len(other_nouns) == 1
        other_noun = other_nouns[0]

        # other_noun = random.choice(other_nouns)
        distractors.append(f"{other_noun} {rel} {obj}")
        distractors.append(f"{subj} {rel} {other_noun}")
        texts = [self.ims_labels[idx][1]] + distractors

        # shuffle the texts and return the label of the correct text
        indices = list(range(len(texts)))
        random.shuffle(indices)
        texts = [texts[i] for i in indices]
        label = indices.index(0)

        return image, texts, label
