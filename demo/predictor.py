# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

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

    def run_on_image(self, image, path, annotation):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # path1 = path.replace('data/','mask/').replace('.png','')
        path2 = path.replace('data/','contour/')
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)

            instances_ = Instances(instances.image_size)
            flag = False
            img_contours = np.zeros(image.shape)

            # instance_num = 0
            # for index in range(len(instances)):
            #     score = instances[index].scores[0]
            #     if score > 0.5:
            #         instance_num += 1

            colors = BBOX_COLORS
            # if instance_num >= 16:
            #     colors = [hsv2rgb(_id / (instance_num + 2), 1) for _id in range(instance_num + 2)]


            color_index = 0
            file_name = path.replace('/workspace/data/','').replace('/','_').replace('.png','')
            annotation[file_name] = {
                'bboxes256': [],
                'mask_colors': [],
                'recogn_id': [],
                'contour_colors': [],
            }
            for index in range(len(instances)):
                score = instances[index].scores[0]
                if score > 0.5:
                    mask = torch.squeeze(instances[index].pred_masks).numpy()*255
                    mask = np.array(mask, np.uint8)
                    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # img_contour = np.zeros(image.shape)
                    # cv2.drawContours(img_contour, contours, -1, (255,255,255), thickness=cv2.FILLED)

                    color = colors[color_index]
                    cv2.drawContours(img_contours, contours, -1, color, thickness=cv2.FILLED)
                    # cv2.imwrite(f'{path1}_{index}.png', img_contour)
                    cv2.imwrite(f'{path2}', img_contours)  
                    color_index +=1
                    mask = torch.from_numpy(mask / 255).unsqueeze(0)
                   
                    annotation[file_name]['contour_colors'].append(list(color))

                    if flag == False:
                        instances_ = instances[index]
                        masks = mask
                        flag = True
                    else:
                        instances_ = Instances.cat([instances_, instances[index]])
                        masks = torch.cat((masks, mask), 0)
                    instances_.pred_masks = masks
            vis_output, vis_annotation  = visualizer.draw_instance_predictions(predictions=instances_, path=path, annotation=annotation)
            

        return predictions, vis_output, vis_annotation


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
