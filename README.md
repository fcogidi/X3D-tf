# X3D-tf

This repository contains a tensorflow implementation of [X3D: Expanding Architectures for Efficient Video Recognition](https://arxiv.org/abs/2004.04730). This reproduction work was done as part of the [2020 ML Reproducibility Challenge](https://paperswithcode.com/rc2020).

X3D networks are derived by expanding multiple axes of a tiny 2D image classification network using a stepwise network expansion method.
These networks are reported to achieve high accuracy on video recognition tasks, even in mobile computational regimes.

**To install requirements**:

***Optional***: ```conda create --name x3d-tf tensorflow-gpu```

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --train_label_file=<path_to_train_label_file> --val_label_file=<path_to_validation_label_file> --model_dir=<path_to_model_directory> --config=<path_to_config_file> --num_gpus=1
```

To view all available options and their descriptions, run:

```help
python train.py --help
```

## Evaluation

To evaluate a model, run:

```eval
TODO
```

## Pre-trained Models

Coming soon...

## Results

This implementation achieves the following performance on the video classification task using the Kinetics-400 dataset:

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |  Test  |
| ------------------ |---------------- | -------------- |  ----  |
| X3D-XS             |     TODO        |      TODO      |  TODO  |
| X3D-S              |     TODO        |      TODO      |  TODO  |
| X3D-M              |     TODO        |      TODO      |  TODO  |
| X3D-L              |     TODO        |      TODO      |  TODO  |
| X3D-XL             |     TODO        |      TODO      |  TODO  |

Training and evaluation are [logged on weights & biases](https://wandb.ai/franklinogidi/X3D-tf)

## TODO

- [ ] Train models on Kinetics-400 dataset
- [ ] Train models on the Charades dataset
  - [ ] Add localization head to network
- [ ] Add multigrid training

Contributions are welcome.

## Citation

If you find this work useful, consider citing the original paper as follows:

```BibTeX
@inproceedings{feichtenhofer2020x3d,
  title={X3D: Expanding Architectures for Efficient Video Recognition},
  author={Feichtenhofer, Christoph},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={203--213},
  year={2020}
}
```

## Acknowledgements

I would like to thank Kumara Kahatapitiya for sharing the training and validation sets of the Kinetics-400 dataset.