# X3D-tf

This repository contains a tensorflow implementation of [X3D: Expanding Architectures for Efficient Video Recognition](https://arxiv.org/abs/2004.04730). This reproduction work was done as part of the [2020 ML Reproducibility Challenge](https://paperswithcode.com/rc2020).

X3D networks are derived by expanding multiple axes of a tiny 2D image classification network using a stepwise network expansion method.
These networks are reported to achieve high accuracy on video recognition tasks, even in mobile computational regimes.

## Installation

***Optional***: ```conda create --name x3d-tf tensorflow-gpu```

```setup
pip install -r requirements.txt
```

## Usage

### Data Preparation

#### Option 1 (recommended): Write video files to TFRecord format

```create tfrecord
PYTHONPATH=".:$PYTHONPATH" python datasets/create_tfrecords.py --set <train, val or test> --video_dir path_to_your_data_folder --label_map datasets/kinetics400/label_map.json --output_dir tfrecords/rec --files_per_record 32
```

#### Option 2: Generate a text file of video paths and label

```create label
PYTHONPATH=".:$PYTHONPATH" python datasets/create_label.py --data_dir path_to_your_data_folder --path_to_label_map datasets/kinetics400/label_map.json --output_path datasets/kinetics400/train.txt
```

### Training

To train the model(s) in the paper, run this command:

```train
python train.py --train_file_pattern tfrecords/rec-train* --val_file_pattern tfrecords/rec-val* --model_dir path_to_your_model_folder --config configs/kinetics/X3D_XS.yaml --num_gpus 1 --use_tfrecords
```

To view all available options and their descriptions, run:

```help
python train.py --help
```

### Evaluation

To evaluate a model, run:

```eval
python eval.py --model_folder path_to_your_model_folder --cfg configs/kinetics/X3D-XS/ --test_label_file datasets/kinetics400/test.json --gpus 1
```

## Results

This implementation achieves the following performance on the video classification task using the Kinetics-400 dataset:

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |  Test  |
| ------------------ |---------------- | -------------- |  ----  |
| X3D-XS             |    TODO        |     TODO     |  TODO  |
| X3D-S              |     TODO        |      TODO      |  TODO  |
| X3D-M              |     TODO        |      TODO      |  TODO  |
| X3D-L              |     TODO        |      TODO      |  TODO  |
| X3D-XL             |     TODO        |      TODO      |  TODO  |

Training and evaluation are [logged on weights & biases](https://wandb.ai/franklinogidi/X3D-tf). Pretrained models can
be found in the `models/` folder.

## Roadmap

- [x] Support both reading from TFRecord files and decoding raw video files
- [ ] Train models on Kinetics-400 dataset
- [ ] Train models on the Charades dataset
  - [ ] Add localization head to network
- [ ] Add multigrid training

**Contributions are welcome.**

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
