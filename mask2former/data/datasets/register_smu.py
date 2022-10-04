from detectron2.data.datasets import register_coco_instances
import os

def register_smu(root):
    register_coco_instances("smu", {}, "/workspace/data/coco_smu.json", "/workspace/data/data256/")

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_smu(_root)