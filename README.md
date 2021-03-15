# A TensorFlow Implementation of X3D

This repository contains a tensorflow implementation of [X3D: Expanding Architectures for Efficient Video Recognition](https://arxiv.org/abs/2004.04730).
X3D networks are derived by expanding multiple axes of a tiny 2D image classification network using a stepwise network expansion method.
This allows the networks to achieve good accuracy-to-complexity trade-off on video classification tasks.

## Installation

***Optional***: ```conda create --name x3d-tf tensorflow-gpu```

```setup
pip install -r requirements.txt
```

## Usage

### Data Preparation
The data preparation options provided here were developed and tested for the kinetics-400 dataset, which can be downloaded using [this repo](https://github.com/Showmax/kinetics-downloader). These options expect the following folder/file structure:
```
-- class_name_1
  -- video_1.mp4
  -- video_2.mkv
-- class_name_2
  -- video_1.avi
  -- video_2.mp4
  .
  .
  .
-- class_name_n
  -- video_1.webm
  -- video_2.mp4
```
The options should work on a custom dataset with a similar file structure.

#### Option 1 (recommended): Write video files to TFRecord format

This option decodes the video frames and encodes each of them as JPEGs before serializing and writing the frames to TFRecord files. Using TFRecord files provides prefetching benefits and improves I/O parallelization, which are especially useful when dealing with video dataset. In other words, using this option, as opposed to the option 2, will speed up training time. The major downside to using this option is that it requires more disk space to store the TFRecord files. In the case of the kinetics-400 dataset, the TFRecord files took 1.3 TB of disk space for the training (~235k videos) and validation (~19.8k videos) sets. This is about a 10x increase in the original dataset size. (Note that only frames making up the first 10 seconds of a video are stored in the TFRecord format).

Use the following command to create TFRecord files:
```create tfrecord
PYTHONPATH=".:$PYTHONPATH" python datasets/create_tfrecords.py --set=<train, val or test> --video_dir=path_to_your_data_folder --label_map=datasets/kinetics400/label_map.json --output_dir=tfrecords/rec --videos_per_record=32
```
To verify/visualize the contents of the TFRecord files, use the following command:
```inspect tfrecord
PYTHONPATH=".:$PYTHONPATH" python datasets/inspect_tfrecord.py --cfg_file=configs/kinetics/X3D_M.yaml --label_map_file=datasets/kinetics400/label_map.json --file_pattern=tfrecords/rec-val-* --eval --num_samples=32
```
#### Option 2: Generate a text file of video paths and label

This option is provided in the case where disk space is limited to store TFRecord files. It will generate a file containing lines of strings where each line contains a path to a video file and the corresponding class id for the video (e. g. `path/to/video.mp4 6`).
```create label
PYTHONPATH=".:$PYTHONPATH" python datasets/create_label.py --data_dir=path_to_your_data_folder --path_to_label_map=datasets/kinetics400/label_map.json --output_path=datasets/kinetics400/train.txt
```

### Training

To train the model(s) in the paper, run this command:

```train
python train.py --train_file_pattern=tfrecords/rec-train* --val_file_pattern=tfrecords/rec-val* --use_tfrecords --pretrained_ckpt=experiments/X3D-XS/ --model_dir=path_to_your_model_folder --config=configs/kinetics/X3D_XS.yaml --num_gpus=1 
```

### Evaluation

To evaluate a model, run:

```eval
python eval.py --model_folder=path_to_your_model_folder --cfg=configs/kinetics/X3D-XS/ --test_label_file=datasets/kinetics400/test.json --gpus=1 --tfrecord
```

## Results

This implementation achieves the following performance on the video classification task using the Kinetics-400 dataset:

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |  Test  |
| ------------------ |---------------- | -------------- |  ----  |
| X3D-XS             |     TODO        |       TODO     |  TODO  |
| X3D-S              |     TODO        |      TODO      |  TODO  |

Training and evaluation are [logged on weights & biases](https://wandb.ai/franklinogidi/X3D-tf).

## Roadmap

- [x] Support both reading from TFRecord files and decoding raw video files
- [ ] Train models on Kinetics-400 dataset
  - [ ] X3D-XS
  - [ ] X3D-S
- [ ] Add multigrid training
- [ ] Add localization head to network
- [ ] Train models on the Charades dataset

**Contributions are welcome.**

## Citation

If you find this work useful, consider citing the original paper:

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

I would like to thank [Kumara Kahatapitiya](https://github.com/kkahatapitiya) for sharing the training and validation sets of the Kinetics-400 dataset.
