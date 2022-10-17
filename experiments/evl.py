# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
from collections import OrderedDict
import json
import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from evl_predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    base_path = "/workspace/data/"
    # img_list = [base_path + line.replace('\n','') for line in open('../data/mixvegrice_test.txt', 'r').readlines()]
    # img_list = [base_path + line.replace('\n','') for line in open('../data/smu_test.txt', 'r').readlines()]
    img_list = [base_path + line.replace('\n','') for line in open('../data/uec_test.txt', 'r').readlines()]
    confidence_scores = [0.3, 0.4, 0.6, 0.7]
    for confidence_score in confidence_scores:
        global testing_data, images_recall, images_precision, images_f1, images_PQ
        testing_data = OrderedDict()
        images_recall, images_precision, images_f1, images_PQ = [], [], [], []
        print(len(testing_data), len(images_PQ), len(images_f1))
        for path in tqdm.tqdm(img_list):
            gt_mask_path = path.replace('/workspace/data/', 'gt_mask/')
            image_name = path.replace('/workspace/data/','').replace('/','_').replace('.png','')
            img = read_image(path, format="BGR")
            test_data, images_PQ, images_f1, images_recall, images_precision = demo.run_on_image(img, image_name, gt_mask_path, confidence_score)
        print(len(images_PQ), len(images_f1))
        overall_PQ = 100 * sum(images_PQ) / len(images_PQ)
        overall_F1 = 100 * sum(images_f1) / len(images_f1)
        overall_recall_ = 100 * sum(images_recall) / len(images_recall)
        overall_precision_ = 100 * sum(images_precision) / len(images_precision)
        test_data["overall"] = {
            "overall_PQ": overall_PQ,
            "overall_F1": overall_F1,
            "overall_recall": overall_recall_,
            "overall_precision": overall_precision_
        }
        print(overall_PQ)
        print(overall_F1)
        print(overall_recall_)
        print(overall_precision_)

        json_data = json.dumps(test_data)
        with open(f'test_uec_r50_90kits_batch8_result_{confidence_score}.json','w') as f:
            f.write(json_data)
        
        # if args.output:
        #     if os.path.isdir(args.output):
        #         assert os.path.isdir(args.output), args.output
        #         out_filename = os.path.join(args.output, os.path.basename(path))
        #     else:
        #         assert len(args.input) == 1, "Please specify a directory with args.output"
        #         out_filename = args.output
        #     visualized_output.save(out_filename)
        # else:
        #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        #     if cv2.waitKey(0) == 27:
        #         break  # esc to quit