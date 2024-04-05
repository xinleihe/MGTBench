# Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts
# Implementation from https://github.com/ArGintum/GPTID, paper link:
# https://arxiv.org/abs/2306.04723

import numpy as np

from scipy.spatial.distance import cdist
from threading import Thread

MINIMAL_CLOUD = 47


def prim_tree(adj_matrix, alpha=1.0):
    infty = np.max(adj_matrix) + 10

    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty

        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)

    return s.item()


def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')


class PHD():
    def __init__(
            self,
            alpha=1.0,
            metric='euclidean',
            n_reruns=3,
            n_points=7,
            n_points_min=3):
        '''
        Initializes the instance of PH-dim computer
        Parameters:
            1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than
        the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
            2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
            3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
            4) n_points --- Number of subsamples to be drawn at each subsample
            5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        '''
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points

            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                reruns[i] = prim_tree(
                    cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)
        outp[thread_id] = (N * (x * y).sum() - x.sum() *
                           y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)

    def fit_transform(
            self,
            X,
            y=None,
            min_points=50,
            max_points=512,
            point_jump=40):
        '''
        Computing the PH-dim
        Parameters:
            1) X --- point cloud of shape (n_points, n_features),
            2) y --- fictional parameter to fit with Sklearn interface
            3) min_points --- size of minimal subsample to be drawn
            4) max_points --- size of maximal subsample to be drawn
            5) point_jump --- step between subsamples
        '''
        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            threads.append(
                Thread(
                    target=self._calc_ph_dim_single,
                    args=[
                        X,
                        test_n,
                        ms,
                        i]))
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        m = np.mean(ms)
        return 1 / (1 - m)
