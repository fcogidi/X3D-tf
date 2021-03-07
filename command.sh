#!/bin/bash
PYTHONPATH=".:$PYTHONPATH" python datasets/create_tfrecords.py --video_dir datasets/kinetics400/val_256/ \
--label_map datasets/kinetics400/label_map.json --output_dir tfrecords/kin400 --set val --files_per_record 64 \
&& PYTHONPATH=".:$PYTHONPATH" python datasets/create_tfrecords.py --video_dir datasets/kinetics400/train_256/ \
--label_map datasets/kinetics400/label_map.json --output_dir tfrecord/kin400 --set train --files_per_record 64
