from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from constants import project_dir, data_dir
from utils_.misc import load_image

cache_image = {}
cache_label = {}


class TCPFewShot(Dataset):
    def __init__(self, opt, split, shot, query,
                 data_root=None, data_list=None, transform=None, mode='train',
                 cache=True):
        assert mode in ['train', 'val', 'test', 'eval_online']

        if mode != "train" and opt.dataset in ["TCP"]:
            mode = "val"
        self.opt = opt
        self.mode = mode
        self.split = split
        self.shot = shot
        self.query = query
        self.data_root = data_root
        self.transform = transform
        self.cache = cache

        self.nclass = 2
        self.data_list = [Path(x) for x in (project_dir / 'data/tcp/tcp.txt').read_text().splitlines()]
        test_range = slice(*opt.test_range)
        self.data_list = self.data_list[test_range]
        self.length_data_list = len(self.data_list)
        self.tasks = list(range(len(self)))
        self.support_image_paths = [list(map(Path, x.split())) for x in (project_dir / 'data/tcp/support.txt').read_text().splitlines()]

        self.sampler = None
        self.reset_sampler()

    def __len__(self):
        return self.opt.test_n or self.length_data_list

    def reset_sampler(self):
        seed = self.opt.seed
        test_seed = self.opt.test_seed
        # Use fixed test sampler(opt.test_seed) for reproducibility
        self.sampler = np.random.RandomState(test_seed)

    def sample_tasks(self):
        pass

    def seg_encode(self, lab, cls, ignore_lab):
        if self.opt.proc == 'pil':
            lab = np.array(lab, np.uint8)
        if len(lab.shape) == 3:
            lab = lab[:, :, -1]
        assert len(lab.shape) == 2, lab.shape
        target_pix = np.where(lab == cls)
        lab[:, :] = 0
        if target_pix[0].shape[0] > 0:
            lab[target_pix[0], target_pix[1]] = 1
        if self.opt.proc == 'pil':
            lab = Image.fromarray(lab)
        return lab

    def get_image(self, name, cache=True):
        if self.cache and cache:
            if name not in cache_image:
                cache_image[name] = load_image(data_dir / name, 'img', self.opt.proc)
            return cache_image[name].copy()
        else:
            return load_image(data_dir / name, 'img', self.opt.proc)

    def get_label(self, name, cls, ignore_lab=0, cache=True):
        if self.cache and cache:
            if name not in cache_label:
                cache_label[name] = load_image(data_dir / name, 'lab', self.opt.proc)
            lab = cache_label[name].copy()
        else:
            lab = load_image(data_dir / name, 'lab', self.opt.proc)
        lab = self.seg_encode(lab, cls, ignore_lab)
        return lab

    def __getitem__(self, index):
        qry_idx = self.tasks[index]
        image_path = self.data_list[qry_idx]
        sup_image_paths = []
        sup_label_paths = []
        for x_path, y_path in self.support_image_paths:
            sup_image_paths.append(x_path)
            sup_label_paths.append(y_path)

        qry_names = [image_path.stem]
        sup_names = [x.stem for x in sup_image_paths]

        ori_image = image = self.get_image(image_path)
        label = np.zeros(image.shape[:2], dtype=np.uint8)
        ori_image = torch.from_numpy(ori_image)

        kwargs = {}
        sup_kwargs = [{} for _ in range(self.opt.shot)]
        sup_images = [self.get_image(x) for x in sup_image_paths]
        sup_labels = [self.get_label(x, 255) for x in sup_label_paths]

        # raw_label = label.copy()
        if self.transform is not None:
            image, label, kwargs = self.transform[1](image, label, **kwargs)
            for k in range(self.shot):
                sup_images[k], sup_labels[k], sup_kwargs[k] = self.transform[0](sup_images[k], sup_labels[k],
                                                                                **sup_kwargs[k])

        sup_images = torch.stack(sup_images, dim=0)
        sup_labels = torch.stack(sup_labels, dim=0)

        ret_dict = {
            'sup_rgb': sup_images,  # [S, 3, H, W]
            'sup_msk': sup_labels,  # [S, H, W]
            'qry_rgb': image,       # [3, H, W]
            'qry_ori': ori_image,   # [H_ori, W_ori, 3]
            'cls': 255,  # [], values in [1, 20] for PASCAL
            'sup_names': sup_names,
            'qry_names': qry_names,
        }

        ret_dict = {k: v for k, v in ret_dict.items() if v is not None}
        return ret_dict
