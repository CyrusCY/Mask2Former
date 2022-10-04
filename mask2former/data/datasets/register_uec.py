from detectron2.data.datasets import register_coco_instances
import os

def register_uec(root):
    register_coco_instances("uec", {}, "/workspace/data/coco_uec.json", "/workspace/data/data256/")

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_uec(_root)