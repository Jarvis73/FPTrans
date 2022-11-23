# Data & Pre-trained Models

Please download the following datasets and pretrained models, and put them into the specified directory.

## Preparing datasets

* [x] PASCAL-5i
* [x] COCO-20i

### PASCAL-5i

* Download [Training/Validation data (2G, tarball)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), and extract `VOCtrainval_11-May-2012.tar` to `./data/`
* Precompute cross-entropy weights
  
  ```bash
  # Dry run to ensure the output path are correct.
  cuda 0 python tools.py precompute_loss_weights with dataset=PASCAL dry_run=True
  # Then generate and save to disk.
  cuda 0 python tools.py precompute_loss_weights with dataset=PASCAL
  ```

### COCO-20i

* Create directory `./data/COCO`
* Download [2014 Training images (13GB, zip)](http://images.cocodataset.org/zips/train2014.zip), [2014 Val images (6GB, zip)](http://images.cocodataset.org/zips/val2014.zip), [2014 Train/Val annotations (241M, zip)](http://images.cocodataset.org/annotations/annotations_trainval2014.zip), and extract them to `./data/COCO/` 
* Generate offline labels

  ```bash
   python tools.py gen_coco_labels with sets=train2014
   python tools.py gen_coco_labels with sets=val2014
  ```

* Precompute cross-entropy weights

  ```bash
  cuda 0 python tools.py precompute_loss_weights with dataset=COCO save_byte=True
  ```

Final directory structure (only display used directories and files):

```
./data
├── COCO
│         ├── annotations
│         ├── train2014
│         ├── train2014_labels
│         ├── val2014
│         ├── val2014_labels
│         └── weights
├── VOCdevkit
│   └── VOC2012
│       ├── SegmentationClassAug
│       ├── JPEGImages
│       └── weights
└── README.md
```
