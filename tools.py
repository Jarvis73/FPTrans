from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sacred import Experiment
from scipy.ndimage import distance_transform_edt
import cv2

from config import MapConfig
from data_kits.datasets import DATA_DIR, DATA_LIST
from utils_.misc import load_image

ex = Experiment("Tools", save_git_info=False, base_dir="./")


@ex.config
def config():
    dataset = "PASCAL"
    proc = "cv2"
    sigma = 5
    save_byte = False       # save weights in uint8 format for saving space (especially for COCO)
    weights_save_dir = DATA_DIR[dataset] / "weights"
    dry_run = False


def boundary2weight(target, cls, kernel, sigma=5):
    mask = torch.zeros(target.shape, dtype=torch.float32).cuda()
    mask[target == cls] = 1
    mask.unsqueeze_(dim=0)      # [1, H, W]
    mask.unsqueeze_(dim=0)      # [1, 1, H, W]

    # Extract mask boundary (inner and outer)
    dilated = torch.clamp(F.conv2d(mask, kernel, padding=1), 0, 1) - mask
    erosion = mask - torch.clamp(F.conv2d(mask, kernel, padding=1) - 8, 0, 1)
    boundary = (dilated + erosion).squeeze(dim=0).squeeze(dim=0)     # [H, W]

    bool_boundary = np.around(boundary.cpu().numpy()).astype(bool)
    edt = distance_transform_edt(np.bitwise_not(bool_boundary))
    weight = np.exp(-edt / sigma ** 2).astype(np.float32)
    return weight


@ex.command(unobserved=True)
def precompute_loss_weights(_config):
    """

    Precompute weights for weighted cross-entropy loss.

    Parameters
    ----------
    _config: ReadOnlyDict
        dataset: str
            Name of the dataset. [PASCAL|COCO]
        save_bytes: bool
            Save weights in float32 or byte. It should be set as True when generating
            for COCO. Default value is False.
    Returns
    -------

    Usage
    -----
    cuda 0 python tools.py precompute_loss_weights with dataset=PASCAL
    cuda 0 python tools.py precompute_loss_weights with dataset=COCO save_bytes=True

    """
    opt = MapConfig(_config)
    save_dir = opt.weights_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = DATA_DIR[opt.dataset]

    # kernel for finding mask boundaries
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float).cuda()

    data_list = DATA_LIST[opt.dataset]['train']
    label_paths = [x.split()[1] for x in data_list.read_text().splitlines()]
    gen = tqdm(label_paths)
    for lab_path in gen:
        save_file = save_dir / Path(lab_path).stem
        gen.set_description(f"{save_file}.npz")
        if save_file.with_suffix('.npz').exists():
            continue

        classes = []
        all_class_edts = []

        lab_path = data_dir / lab_path
        label = load_image(lab_path, 'lab', opt.proc)
        unique_labels = np.unique(label).tolist()

        for cls in unique_labels:
            if cls == 255:
                continue
            classes.append(cls)
            all_class_edts.append(boundary2weight(label, cls, kernel, opt.sigma))

        classes = np.array(classes)
        edt = np.stack(all_class_edts, axis=0)
        if opt.save_byte:
            edt = (edt * 255).astype('uint8')

        if not opt.dry_run:
            np.savez_compressed(save_file, x=edt, c=classes)


@ex.command(unobserved=True)
def print_ckpt(ckpt):
    """

    This tool helps print the weight names and shapes for inspecting a checkpoint.

    Parameters
    ----------
    ckpt: str
        Path to a checkpoint

    """
    state = torch.load(ckpt, map_location='cpu')
    if 'model_state' in state:
        state = state['model_state']
    elif 'state_dict' in state:
        state = state['state_dict']
    elif 'model' in state:
        state = state['model']

    max_name_length = max([len(x) for x in state])
    max_shape_length = max([len(str(x.shape)) for x in state.values()])
    pattern = "  {:<%ds}  {:<%ds}" % (max_name_length, max_shape_length)

    print_str = ""
    for k, v in state.items():
        print_str += pattern.format(k, str(list(v.shape))) + "\n"

    print(print_str)


@ex.command(unobserved=True)
def gen_coco_labels(sets, _config):
    """

    Generate COCO labels with 'pycocotools' API.

    Parameters
    ----------
    sets: str
        Data sets. The accessible values are [train2014, val2014].
    _config: ReadOnlyDict
        dry_run: bool
            Dry run this command without saving to disk.

    Returns
    -------

    Usage
    -----
    python tools.py gen_coco_labels with sets=train2014
    python tools.py gen_coco_labels with sets=val2014

    """
    from pycocotools.coco import COCO

    opt = MapConfig(_config)
    if sets not in ['train2014', 'val2014']:
        raise ValueError(f'Not supported sets: {sets}. [train2014, val2014]')
    save_dir = DATA_DIR['COCO'] / f'{sets}_label'
    annFile = DATA_DIR['COCO'] / f'annotations/instances_{sets}.json'
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'Labels of {sets} are saved to {save_dir}.')

    coco = COCO(str(annFile))
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())

    nms = [cat['name'] for cat in cats]
    num_cats = len(nms)
    print('All {} categories.'.format(num_cats))
    print(nms)

    # get all images ids
    imgIds = coco.getImgIds()
    gen = tqdm(enumerate(imgIds), total=len(imgIds))
    for idx, im_id in gen:
        # load annotations
        annIds = coco.getAnnIds(imgIds=im_id, iscrowd=False)
        if len(annIds) == 0:
            continue

        image = coco.loadImgs([im_id])[0]
        # image.keys: ['coco_url', 'flickr_url', 'date_captured', 'license', 'width', 'height', 'file_name', 'id']
        h, w = image['height'], image['width']
        gt_name = image['file_name'].split('.')[0] + '.png'
        gt = np.zeros((h, w), dtype=np.uint8)

        # ann.keys: ['area', 'category_id', 'bbox', 'iscrowd', 'id', 'segmentation', 'image_id']
        anns = coco.loadAnns(annIds)
        for ann_idx, ann in enumerate(anns):

            cat = coco.loadCats([ann['category_id']])
            cat = cat[0]['name']
            cat = nms.index(cat) + 1  # cat_id ranges from 1 to 80

            # below is the original script
            segs = ann['segmentation']
            for seg in segs:
                seg = np.array(seg).reshape(-1, 2)  # [n_points, 2]
                cv2.fillPoly(gt, [seg.astype(np.int32)], int(cat))

        save_gt_path = save_dir / gt_name
        gen.set_description(f'{save_gt_path}')
        if not opt.dry_run:
            cv2.imwrite(str(save_gt_path), gt)


if __name__ == "__main__":
    ex.run_commandline()
