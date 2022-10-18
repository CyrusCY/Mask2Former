# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque
from turtle import goto
from PIL import Image
from collections import OrderedDict
import json

import cv2
import torch
import numpy as np

from detectron2.structures import Instances
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from utils.visualizer import ColorMode, Visualizer


DARK_RED = (136, 0, 21)
RED = (237, 28, 36)
ORANGE = (255, 127, 39)
YELLOW = (255, 242, 0)
GREEN = (34, 177, 76)
TURQUOISE = (0, 162, 232)
INDIGO = (63, 72, 204)
PURPLE = (163, 73, 164)
BROWN = (185, 122, 87)
ROSE = (255, 174, 201)
GOLD = (255, 201, 14)
LIGHT_YELLOW = (239, 228, 176)
LIME = (181, 230, 29)
LIGHT_TURQUOISE = (153, 217, 234)
BLUE_GRAY = (112, 146, 190)
LAVENDER = (200, 191, 231)

BBOX_COLORS = [DARK_RED, RED, ORANGE, YELLOW, GREEN, TURQUOISE, INDIGO, PURPLE,
               BROWN, ROSE, GOLD, LIGHT_YELLOW, LIME, LIGHT_TURQUOISE, BLUE_GRAY, LAVENDER]

import math
import colorsys

def color(c):
    return int(math.floor(c * 255))
def hsv2rgb(h, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, 1.0, v)
    return (color(r), color(g), color(b))

SIZE = 256
testing_data = OrderedDict()

images_recall, images_precision, images_f1, images_PQ = [], [], [], []

metadata = {}
with open('metadata.json','r') as json_file:
    metadata = json.load(json_file)

def get_binary_mask(instance_mask, color):
    from scipy import ndimage
    binary_color_mask = (instance_mask[:, :, 0] == color[0])
    binary_color_mask = np.logical_and(binary_color_mask, (instance_mask[:, :, 1] == color[1]))
    binary_color_mask = np.logical_and(binary_color_mask, (instance_mask[:, :, 2] == color[2]))
    binary_color_mask = ndimage.binary_opening(binary_color_mask)
    return binary_color_mask

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, image_name, gt_mask_path, confidence_score):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        global testing_data, images_recall, images_precision, images_f1, images_PQ

        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        # visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        gt_im = Image.open(gt_mask_path)
        gt_rgb_im = gt_im.convert('RGB')
        gt_rgb_mask = np.array(gt_rgb_im)
        testing_data[image_name] = OrderedDict()
        gt_image_data = metadata[image_name]
        gt_instance_mask = np.zeros((SIZE, SIZE))
        gt_num_instances = len(gt_image_data['recogn_id'])
        for bid in range(gt_num_instances):
            binary_mask = get_binary_mask(gt_rgb_mask, gt_image_data['mask_colors'][bid])
            gt_instance_mask[binary_mask] = bid + 1
        
        semantic_performance_ = []
        gt_ids = list(np.unique(gt_instance_mask))
        gt_ids.sort()
        gt_ids = gt_ids[1:]

        gt_pred_pairs = []
        pred_id = 0
        recogn_ids = []
        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)

            for index in range(len(instances)):
                score = instances[index].scores[0]
                if score > confidence_score:
                    pred_id += 1

                    recogn_ids.append(instances[index].pred_classes.tolist()[0]+1)

                    mask = torch.squeeze(instances[index].pred_masks).numpy()*255
                    mask = np.array(mask, np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img_contours = np.zeros(image.shape)
                    cv2.drawContours(img_contours, contours, -1, (255,255,255), thickness=cv2.FILLED)
                    cv2.imwrite(f'temp.png', img_contours)
                    mask = torch.from_numpy(mask / 255).unsqueeze(0)
                    im = Image.open('temp.png')
                    im = im.convert('1') 
                    img_contours = np.array(im)
                    for gt_id_ in gt_ids:
                        gt_binary_mask = np.zeros((SIZE, SIZE))
                        gt_binary_mask[gt_instance_mask == gt_id_] = 1
                        
                        tp = np.logical_and(gt_binary_mask, img_contours)
                        union = np.sum(gt_binary_mask) + np.sum(img_contours) - np.sum(tp)
                        iou = 0 if union == 0 else np.sum(tp) / np.sum(union)

                        gt_pred_pairs.append((iou, int(gt_id_), int(pred_id)))                        
            
            gt_pred_pairs.sort(reverse=True)
            gt_, pred_ = OrderedDict(), OrderedDict()
            for iou, gt_id, pred_id in gt_pred_pairs:
                gt_recogn_ = gt_image_data['recogn_id'][gt_id-1]
                pred_recogn_ = recogn_ids[pred_id-1]
                if iou == 0: continue
                if (gt_id in gt_ and pred_id in pred_): continue
                if (gt_id in gt_ or pred_id in pred_) and (gt_recogn_ != pred_recogn_): continue
                semantic_performance_.append((iou, gt_recogn_, pred_recogn_))
                gt_[gt_id] = True
                pred_[pred_id] = True

            for iou, gt_id, pred_id in gt_pred_pairs:
                if gt_id not in gt_:
                    semantic_performance_.append((0, gt_image_data['recogn_id'][gt_id-1], 0)) #false negative
                    gt_[gt_id] = False

                if pred_id not in pred_:
                    semantic_performance_.append((0, 0, recogn_ids[pred_id-1])) #false positive
                    pred_[pred_id] = False

            tp, sum_iou = 0, 0
            recall_, precision_ = [], []
            for iou, gt, pred in semantic_performance_:
                if gt == pred:
                    tp += 1
                    sum_iou += iou
                if gt != 0:
                    recall_.append(int(gt == pred))
                if pred != 0:
                    precision_.append(int(gt == pred))
            pq_ = sum_iou / max(len(semantic_performance_) / 2 + tp / 2, 1)
            testing_data[image_name]['semantic_performance'] = semantic_performance_
            testing_data[image_name]['recall'] = recall_
            testing_data[image_name]['precision'] = precision_
            testing_data[image_name]['pq'] = pq_

            # overall_iou += sum_iou
            # overall_tp += tp
            # overall_num_instances += len(semantic_performance_)

            images_PQ.append(pq_)
            precision__ = sum(precision_) / max(len(precision_), 1)
            recall__ = sum(recall_) / max(len(recall_), 1)
            f1__ = 2 * precision__ * recall__ / max((precision__ + recall__), 1)
            images_recall.append(recall__)
            images_precision.append(precision__)
            testing_data[image_name]['f1'] = f1__
            images_f1.append(f1__)
            print(f'{confidence_score} - {image_name} PQ: {pq_} F1: {f1__}')
            return testing_data, images_PQ, images_f1, images_recall, images_precision


        # return predictions, vis_output, vis_annotation


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5