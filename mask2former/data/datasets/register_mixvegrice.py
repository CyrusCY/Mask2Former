from detectron2.data.datasets import register_coco_instances
import os

def register_mixvegrice(root):
    register_coco_instances("mixvegrice", {}, "/workspace/data/coco_mixvegrice.json", "/workspace/data/data256/")

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_mixvegrice(_root)