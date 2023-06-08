DATASET_DIR = "data/datasets/"

DATASET_PATHS = {
    "single-object": {
        "train_image_path": DATASET_DIR + "single_object/images/",
        "val_image_path": DATASET_DIR + "single_object/images/",
        "gen_image_path": DATASET_DIR + "single_object/images/",
        "train_label_path": DATASET_DIR + "single_object/train.csv",
        "val_label_path": DATASET_DIR + "single_object/val.csv",
        "gen_label_path": DATASET_DIR + "single_object/test.csv",
    },
    "rel": {
        "train_image_path": DATASET_DIR + "rel/images/train/",
        "val_image_path": DATASET_DIR + "rel/images/val/",
        "gen_image_path": DATASET_DIR + "rel/images/gen/",
        "train_label_path": DATASET_DIR + "rel/train.json",
        "val_label_path": DATASET_DIR + "rel/val.json",
        "gen_label_path": DATASET_DIR + "rel/gen.json",
    },
    "two-object": {
        "train_image_path": DATASET_DIR + "two_object/images/train/",
        "val_image_path": DATASET_DIR + "two_object/images/val/",
        "gen_image_path": DATASET_DIR + "two_object/images/gen/",
        "train_label_path": DATASET_DIR + "two_object/train.csv",
        "val_label_path": DATASET_DIR + "two_object/val.csv",
        "gen_label_path": DATASET_DIR + "two_object/gen.csv",
    },
}
