# Defect detection
we demonstrate how to use two types of Amazon Sagemaker built-in OD models to finetune on the Steel Surface Defect dataset, which is used in this solution.

Type 1 (legacy): uses a built-in legacy Object Detection algorithm and uses the Single Shot multibox Detector (SSD) model with either VGG or ResNet backbone, and was pretrained on the ImageNet dataset.
Type 2 (latest): provides 9 pretrained OD models, including 8 SSD models and 1 FasterRCNN model. These models use VGG, ResNet, or MobileNet as backbone, and were pretrained on COCO, VOC, or FPN datasets.
# License
MIT
