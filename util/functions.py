import warnings

import numpy
import scipy.sparse.csr
import torch


def get_density(M):
    r, c = M.shape
    return len(M.indices) / (c * r)


def get_row(M, index: int) -> list:
    if type(M) == numpy.ndarray:
        return M[index]
    return M.indices[M.indptr[index]: M.indptr[index + 1]]


def get_col(M: scipy.sparse.csc.csc_matrix, index: int) -> list:
    return M.indices[M.indptr[index]: M.indptr[index + 1]]


def mix_row(row_list) -> list:
    rst = []
    for r in row_list:
        rst = rst + list(r)
    rst = list(set(rst))
    rst = sorted(rst)
    return rst
