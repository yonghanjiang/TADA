import itertools

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from groundingdino.datasets import DetrDatasetMapper, MetadataCatalog, DatasetCatalog

root = "datasets/odinw13"

dataloader = OmegaConf.create()

if "odinw13_test" not in DatasetCatalog.list():
    register_coco_instances("odinw13_test", {},
                            f"{root}/merged_annotations.json",
                            f"{root}")
DatasetCatalog.get("odinw13_test")
test_thing_classes = MetadataCatalog.get("odinw13_test").thing_classes


dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="odinw13_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
        categories_names=test_thing_classes,
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
