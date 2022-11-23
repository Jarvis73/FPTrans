import numpy as np


class FewShotMetric(object):
    def __init__(self, n_class):
        self.n_class = n_class
        self.stat = np.zeros((self.n_class + 1, 3))     # +1 for bg, 3 for tp, fp, fn

    def update(self, pred, ref, cls, ori_size=None, verbose=0):
        pred = np.asarray(pred, np.uint8)
        ref = np.asarray(ref, np.uint8)
        for i, ci in enumerate(cls):    # iter on batch ==> [episode_1, episode_2, ...]
            p = pred[i]
            r = ref[i]
            if ori_size is not None:
                ori_H, ori_W = ori_size[i]
                p = p[:, :ori_H, :ori_W]
                r = r[:, :ori_H, :ori_W]
            for j, c in enumerate([0, int(ci)]):     # iter on class ==> [bg_cls, fg_cls]
                tp = int((np.logical_and(p == j, r != 255) * np.logical_and(r == j, r != 255)).sum())
                fp = int((np.logical_and(p == j, r != 255) * np.logical_and(r != j, r != 255)).sum())
                fn = int((np.logical_and(p != j, r != 255) * np.logical_and(r == j, r != 255)).sum())
                if verbose:
                    print(tp / (tp + fp + fn))
                self.stat[c, 0] += tp
                self.stat[c, 1] += fp
                self.stat[c, 2] += fn

    def get_scores(self, labels, binary=False):
        """
        Parameters
        ----------
        labels: list, compute mean iou on which classes
        binary: bool, return mean iou (average on foreground classes) 
            or binary iou (average on foreground and background)

        Returns
        -------
        mIoU_class: np.ndarray, all ious of foreground classes
        mean: float, mean iou over foreground classes
        """
        if binary:
            stat = np.c_[self.stat[0], self.stat[1:].sum(axis=0)].T     # [2, 3]
        else:
            stat = self.stat[labels]                                    # [N, 3]
        
        tp, fp, fn = stat.T                                             # [2 or N]
        mIoU_class = tp / (tp + fp + fn)                                # [2 or N]
        mean = np.nanmean(mIoU_class)                                   # scalar

        return mIoU_class, mean

    def reset(self):
        self.stat = np.zeros((self.n_class + 1, 3))     # +1 for bg, 3 for tp, fp, fn


class SegmentationMetric(object):
    def __init__(self, n_class):
        self.n_class = n_class
        self.confusion_matrix = np.zeros((n_class, n_class))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, pred, ref, ori_size=None):
        pred = np.asarray(pred, np.uint8)
        ref = np.asarray(ref, np.uint8)
        for i, (r, p) in enumerate(zip(ref, pred)):
            if ori_size is not None:
                ori_H, ori_W = ori_size[i]
                p = p[:ori_H, :ori_W]
                r = r[:ori_H, :ori_W]
            self.confusion_matrix += self._fast_hist(
                r.flatten(), p.flatten(), self.n_class
            )

    def get_scores(self, binary=False, withbg=True):
        if not binary:
            hist = self.confusion_matrix
        else:
            hist = np.zeros((2, 2), dtype=np.int32)
            hist[0, 0] = self.confusion_matrix[0, 0]
            hist[0, 1] = self.confusion_matrix[0, 1:].sum()
            hist[1, 0] = self.confusion_matrix[1:, 0].sum()
            hist[1, 1] = self.confusion_matrix[1:, 1:].sum()
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        if not withbg:
            iou = iou[1:]
        mean_iou = np.nanmean(iou)

        return iou, mean_iou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_class, self.n_class))


class Accumulator(object):
    def __init__(self, **kwargs):
        self.values = kwargs
        self.counter = {k: 0 for k, v in kwargs.items()}
        for k, v in self.values.items():
            if not isinstance(v, (float, int, list)):
                raise TypeError(f"The Accumulator does not support `{type(v)}`. Supported types: [float, int, list]")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(self.values[k], list):
                self.values[k].append(v)
            else:
                self.values[k] = self.values[k] + v
            self.counter[k] += 1

    def reset(self):
        for k in self.values.keys():
            if isinstance(self.values[k], list):
                self.values[k] = []
            else:
                self.values[k] = 0
            self.counter[k] = 0

    def mean(self, key, axis=None, dic=False):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).mean(axis)
            else:
                return self.values[key] / (self.counter[key] + 1e-7)
        elif isinstance(key, (list, tuple)):
            if dic:
                return {k: self.mean(k, axis) for k in key}
            return [self.mean(k, axis) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")

    def std(self, key, axis=None, dic=False):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).std(axis)
            else:
                raise RuntimeError("`std` is not supported for (int, float). Use list instead.")
        elif isinstance(key, (list, tuple)):
            if dic:
                return {k: self.std(k) for k in key}
            return [self.std(k) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")
