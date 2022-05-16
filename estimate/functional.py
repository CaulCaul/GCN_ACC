import warnings

import scipy.sparse.csr
import torch


def get_density(M):
    r, c = M.shape
    return len(M.indices) / (c * r)


def mix_row(row_list):
    rst = []
    for r in row_list:
        rst = rst + list(r)
    rst = list(set(rst))
    rst = sorted(rst)
    return rst

