"""
A transformation package for image segmentation. It is implemented by PIL and
torchvision and its usage is the same as torchvision, except for all the
classes accept (img, mask) as input and output (img, mask).

Notes:
Opencv(cv2)-processed images and PIL-processed images may have some differences,
which may lead to incompatibility between the images and pretrained models and
performance drop.
"""

import torchvision.transforms as transforms
import torchvision.transforms.functional as tf
import numpy as np
from PIL import Image, ImageFilter
import random
import torch
import cv2

__all__ = [
    "Compose",
    "Resize", "ToTensor", "Normalize",
    "ColorJitter", "RandomCrop", "RandomResize", "RandomSizedCrop",
    "RandomHorizontallyFlip", "RandomVerticallyFlip", "RandomRotate",
    "RandomGaussianBlur"
]
RGB_MEAN = np.array([0.485, 0.456, 0.406]) * 255


class Compose(object):
    def __init__(self, transformations, processer='pil'):
        self.transforms = transformations
        self.processer = processer

    def __call__(self, image, label, **kwargs):
        if self.processer == 'cv2':
            image = np.float32(image)
            label = np.uint8(label)
        for t in self.transforms:
            image, label, kwargs = t(image, label, proc=self.processer, **kwargs)
            
        return image, label, kwargs


###############################################################################
#
#   Deterministic Algorithms
#
###############################################################################

class Resize(object):
    """
    Resize a pair of (image, mask).
    If `height` and `width` are both specified, then the aspect ratio is not
    keeped. If only one of the `height` and `width` is specified, the aspect
    ratio will be preserved.

    Parameters
    ----------
    height: int
    width: int
    do_mask: bool
        Do resize on mask or not.
    """
    def __init__(self, height=None, width=None, do_mask=True):
        if height is None and width is None:
            raise ValueError('At least specify one of the `height` and '
                             '`width`')
        self.height = height
        self.width = width
        self.do_mask = do_mask

    def __call__(self, img, mask, proc='pil', **kwargs):
        if proc == 'pil':
            w, h = img.size
        else:
            h, w, _ = img.shape
        ow, oh = self.width, self.height
        if self.height is None:
            oh = int(self.width * h / w)
        if self.width is None:
            ow = int(self.height * w / h)

        if proc == 'pil':
            img = img.resize((ow, oh), Image.BILINEAR)
            if self.do_mask:
                mask = mask.resize((ow, oh), Image.NEAREST)
        else:
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
            if self.do_mask:
                mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

        if 'weights' in kwargs:
            kwargs['weights'] = cv2.resize(kwargs['weights'], (ow, oh), interpolation=cv2.INTER_LINEAR)

        return img, mask, kwargs


class ToTensor(object):
    def __init__(self, mask_dtype):
        self.mask_dtype = mask_dtype

    def __call__(self, img, mask, proc='pil', **kwargs):
        img = torch.from_numpy(
            np.ascontiguousarray(np.asarray(img).transpose(2, 0, 1))
        ).float().div(255)

        mask = torch.from_numpy(np.array(mask, np.uint8))
        if self.mask_dtype == 'long':
            mask = mask.long()
        elif self.mask_dtype == 'float':
            mask = mask.float()

        if 'weights' in kwargs:
            kwargs['weights'] = torch.from_numpy(kwargs['weights']).float()

        return img, mask, kwargs


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask, proc='pil', **kwargs):
        img = tf.normalize(img, self.mean, self.std)

        return img, mask, kwargs


###############################################################################
#
#   Random Algorithms
#
###############################################################################

class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(
            brightness, contrast, saturation, hue
        )

    def __call__(self, img, mask, proc='pil', **kwargs):
        if random.random() > self.p:
            if proc == 'pil':
                img = self.jitter(img)
            else:
                img = Image.fromarray(img.astype(np.uint8))
                img = self.jitter(img)
                img = np.array(img)

        return img, mask, kwargs


class RandomGaussianBlur(object):
    def __init__(self, radius=5, rand_radius=False):
        self.radius = radius
        self.rand_radius = rand_radius

    def __call__(self, img, mask, proc='pil', **kwargs):
        if random.random() < 0.5:
            if proc == 'pil':
                img = img.filter(ImageFilter.GaussianBlur(self.radius))
            else:
                radius = random.choice([self.radius, self.radius - 2, self.radius + 2]) if self.rand_radius else self.radius
                img = cv2.GaussianBlur(img, (radius, radius), 0)

        return img, mask, kwargs


class RandomCrop(object):
    """
    Random crop a patch from the (img, mask) pair.

    Parameters
    ----------
    height: int
        Output height
    width: int
        Output width
    check: bool
        Whether to check the cropped mask contains a minimum proportion of
        original foreground/background pixels. Only support two classes
    prop: float
        Foreground pixel proportion.
    max_iter: int
        Maximum iterations to try (to meet the `check` requirement).
    """
    def __init__(self, height, width, check=False, prop=0.85, max_iter=30, center=True, pad_type='reflect'):
        self.height = height
        self.width = width
        self.check = check
        self.prop = prop
        self.max_iter = max_iter
        self.center = center
        if pad_type == 'reflect':
            self.pad_kwargs_image = self.pad_kwargs_mask = self.pad_kwargs_weight = {
                'borderType': cv2.BORDER_REFLECT101
            }
        elif pad_type == 'constant':
            self.pad_kwargs_image = {
                'borderType': cv2.BORDER_CONSTANT,
                'value': RGB_MEAN
            }
            self.pad_kwargs_mask = {
                'borderType': cv2.BORDER_CONSTANT,
                'value': 255,
            }
            self.pad_kwargs_weight = {
                'borderType': cv2.BORDER_CONSTANT,
                'value': 0,
            }
        else:
            raise ValueError

    def __call__(self, img, mask, proc='pil', **kwargs):
        if proc == 'pil':
            w, h = img.size
            mw, mh = mask.size
        else:
            h, w, _ = img.shape
            mh, mw = mask.shape
        tw, th = self.width, self.height
        assert w == mw and h == mh, f'(W, H) => ({w}, {h}) vs ({mw}, {mh})'
        weights = kwargs.get('weights', None)

        if proc not in ['pil', 'cv2']:
            raise ValueError

        if w != tw or h != th:
            if w < tw or h < th:
                if proc == 'pil':
                    img = img.resize((tw, th), Image.BILINEAR)
                    mask = mask.resize((tw, th), Image.NEAREST)
                else:
                    # img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                    # mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                    pad_h = max(th - h, 0)
                    pad_w = max(tw - w, 0)
                    if self.center:
                        pad_h_half = int(pad_h / 2)
                        pad_w_half = int(pad_w / 2)
                    else:
                        pad_h_half = random.randint(0, pad_h)
                        pad_w_half = random.randint(0, pad_w)
                    borders = (pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half)
                    if pad_h > 0 or pad_w > 0:
                        img = cv2.copyMakeBorder(img, *borders, **self.pad_kwargs_image)
                        mask = cv2.copyMakeBorder(mask, *borders, **self.pad_kwargs_mask)
                if weights is not None:
                    weights = cv2.copyMakeBorder(weights, *borders, **self.pad_kwargs_weight)

            raw_img = img
            raw_mask = mask
            raw_weights = weights
            if proc == 'pil':
                w, h = img.size
            else:
                h, w, _ = img.shape
            y1 = random.randint(0, h - th)
            x1 = random.randint(0, w - tw)
            if proc == 'pil':
                img = raw_img.crop((x1, y1, x1 + tw, y1 + th))
                mask = raw_mask.crop((x1, y1, x1 + tw, y1 + th))
            else:
                img = raw_img[y1:y1 + th, x1:x1 + tw]
                mask = raw_mask[y1:y1 + th, x1:x1 + tw]

            if weights is not None:
                weights = weights[y1:y1 + th, x1:x1 + tw]

            if self.check:
                # For semgentation, we must assert the cropped patch contains
                # foreground pixels
                np_raw_mask = np.array(raw_mask, np.uint8)
                np_mask = np.array(mask, np.uint8)
                raw_pos_num = np.sum(np_raw_mask == 1)
                pos_num = np.sum(np_mask == 1)
                crop_iter = 0
                while pos_num < self.prop * raw_pos_num and crop_iter <= self.max_iter:
                    image = raw_img
                    label = raw_mask
                    weights = raw_weights
                    y1 = random.randint(0, h - th)
                    x1 = random.randint(0, w - tw)
                    if proc == 'pil':
                        img = raw_img.crop((x1, y1, x1 + tw, y1 + th))
                        mask = raw_mask.crop((x1, y1, x1 + tw, y1 + th))
                    else:
                        img = raw_img[y1:y1 + th, x1:x1 + tw]
                        mask = raw_mask[y1:y1 + th, x1:x1 + tw]
                    if weights is not None:
                        weights = weights[y1:y1 + th, x1:x1 + tw]
                    crop_iter += 1
                if crop_iter >= self.max_iter:
                    if proc == 'pil':
                        img = raw_img.resize((tw, th), Image.BILINEAR)
                        mask = raw_mask.resize((tw, th), Image.NEAREST)
                    else:
                        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                    if weights is not None:
                        weights = cv2.resize(weights, (tw, th), interpolation=cv2.INTER_LINEAR)

        if weights is not None:
            kwargs['weights'] = weights

        return img, mask, kwargs


class RandomResize(object):
    def __init__(self, smin, smax):
        self.smin = smin
        self.smax = smax

    def __call__(self, img, mask, proc='pil', **kwargs):
        if proc == 'pil':
            w, h = img.size
            mw, mh = mask.size
        else:
            h, w, _ = img.shape
            mh, mw = mask.shape
        assert w == mw and h == mh, f'(W, H) => ({w}, {h}) vs ({mw}, {mh})'

        scale = self.smin + (self.smax - self.smin) * random.random()

        tw = int(w * scale)
        th = int(h * scale)

        if proc == 'pil':
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
        else:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        if 'weights' in kwargs:
            kwargs['weights'] = cv2.resize(kwargs['weights'], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        return img, mask, kwargs


class RandomSizedCrop(object):
    def __init__(self, smin, smax, height, width,
                 check=False, fg_prop=0.85, bg_prop=0.15, max_iter=30, center=True, pad_type='reflect'):
        self.resize = RandomResize(smin, smax)
        self.crop = RandomCrop(height, width, check, fg_prop, bg_prop, max_iter, center, pad_type)

    def __call__(self, img, mask, proc='pil', **kwargs):
        img, mask, kwargs = self.resize(img, mask, proc=proc, **kwargs)
        img, mask, kwargs = self.crop(img, mask, proc=proc, **kwargs)
        return img, mask, kwargs


class RandomHorizontallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, proc='pil', **kwargs):
        if random.random() < self.p:
            if proc == 'pil':
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = np.ascontiguousarray(np.flip(img, axis=1))
                mask = np.ascontiguousarray(np.flip(mask, axis=1))
            if 'weights' in kwargs:
                kwargs['weights'] = np.ascontiguousarray(np.flip(kwargs['weights'], axis=1))

        return img, mask, kwargs


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, proc='pil', **kwargs):
        if random.random() < self.p:
            if proc == 'pil':
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                img = np.ascontiguousarray(np.flip(img, axis=0))
                mask = np.ascontiguousarray(np.flip(mask, axis=0))
            if 'weights' in kwargs:
                kwargs['weights'] = np.ascontiguousarray(np.flip(kwargs['weights'], axis=0))

        return img, mask, kwargs


class RandomRotate(object):
    def __init__(self, rotate, p=0.5, pad_type='reflect'):
        self.rotate = rotate
        self.p = p
        if pad_type == 'reflect':
            self.pad_kwargs_image = self.pad_kwargs_mask = self.pad_kwargs_weight = {
                'borderMode': cv2.BORDER_REFLECT101
            }
        elif pad_type == 'constant':
            self.pad_kwargs_image = {
                'borderMode': cv2.BORDER_CONSTANT,
                'borderValue': RGB_MEAN,
            }
            self.pad_kwargs_mask = {
                'borderMode': cv2.BORDER_CONSTANT,
                'borderValue': 255,
            }
            self.pad_kwargs_weight = {
                'borderMode': cv2.BORDER_CONSTANT,
                'borderValue': 0,
            }
        else:
            raise ValueError

    def __call__(self, img, mask, proc='pil', **kwargs):
        if random.random() < self.p:
            angle = -self.rotate + 2 * self.rotate * random.random()

            if proc == 'pil':
                img, mask = np.array(img, np.uint8), np.array(mask, np.uint8)

            h, w, _ = img.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, **self.pad_kwargs_image)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, **self.pad_kwargs_mask)
            if proc == 'pil':
                img, mask = Image.fromarray(img), Image.fromarray(mask)

            if 'weights' in kwargs:
                kwargs['weights'] = cv2.warpAffine(kwargs['weights'], matrix, (w, h), flags=cv2.INTER_LINEAR, **self.pad_kwargs_weight)
        
        return img, mask, kwargs

