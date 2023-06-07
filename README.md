# Does CLIP Bind Concepts?

Code to reproduce the experiments in the paper: [Does CLIP Bind Concepts? Probing Compositionality in Large Image Models](https://arxiv.org/abs/2212.10537).

## Setup
To install the required packages, run:
```
conda create -n clip-binding python=3.9
conda activate clip-binding
mkdir data
pip3 install -r requirements.txt
```

## Dataset
You can download the dataset for all our experiments from [Google Drive](https://drive.google.com/drive/folders/1mFQTaIYIE01fOe1Wvc51V8cGbFuDzGBI?usp=sharing).

Download the dataset, unzip it, and place it in the `data` directory.


## Training
To run the training script, run:
```
python3 train.py --model_name=csp --dataset=single-object
```

You can specify the following arguments:
- `--model_name`: The model to train. One of `clip`, `csp`, `add`, `mult`, `conv`, `tl`, `rf`.
- `--dataset`: The dataset to train on. One of `single-object`, `two-object`, `relational`.
- `--save_dir`: The directory to save the results and intermediate predictions. By default, the save directory is set to `data/<dataset>/<model_name>_seed_0`.

Notes:
1.  `--evaluate_only`: To evaluate pretrained CLIP, set this to `True` and set the `--model_name=clip`.
2.  Change the learning rate to `--lr=1e-07` to fine-tune CLIP and `--lr=1e-05` to train the CDSMs (`add`, `mult`, `conv`, `tl`, `rf`).

## Citation

If you find this code useful, please cite our paper:

```
@article{lewis2022does,
  title={Does CLIP Bind Concepts? Probing Compositionality in Large Image Models},
  author={Lewis, Martha and Yu, Qinan and Merullo, Jack and Pavlick, Ellie},
  journal={arXiv preprint arXiv:2212.10537},
  year={2022}
}
```