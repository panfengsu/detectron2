#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import sys
import yaml
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_traffic_lights_dicts(img_dir):
    yaml_file = os.path.join(img_dir, "train.yaml")
    imgs_anns = yaml.load(open(yaml_file, 'rb').read())

    print(len(imgs_anns))

    dataset_dicts = []
    for idx in range(len(imgs_anns)):

        record = {}
        
        filename = os.path.abspath(os.path.join(os.path.dirname(yaml_file), imgs_anns[idx]['path']))
        try:
            height, width = cv2.imread(filename).shape[:2]
        except:
            print("{} not exist".format(filename))
            continue
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = imgs_anns[idx]["boxes"]
        objs = []
        for anno in annos:

            obj = {
                "bbox": [anno['x_min'], anno['y_min'], anno['x_max'], anno['y_max']],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id":0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train",]:
    DatasetCatalog.register("traffic_lights_" + d, lambda d=d: get_traffic_lights_dicts("/data/traffic_lights_data"))
    MetadataCatalog.get("traffic_lights_" + d).set(thing_classes=["traffic_lights"])
balloon_metadata = MetadataCatalog.get("traffic_lights_train")

'''
dataset_dicts = get_traffic_lights_dicts("/data/traffic_lights_data")
print(dataset_dicts)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("test",out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
'''

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
cfg.DATASETS.TRAIN = ("traffic_lights_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()




  
