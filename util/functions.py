import warnings

import scipy.sparse.csr
import torch


def get_density(M):
    r, c = M.shape
    return len(M.indices) / (c * r)


def get_row(M: scipy.sparse.csr.csr_matrix, index: int):
    # print(M.indptr[index], M.indptr[index + 1])
    return M.indices[M.indptr[index]: M.indptr[index + 1]]


def get_col(M: scipy.sparse.csc.csc_matrix, index: int):
    return M.indices[M.indptr[index]: M.indptr[index + 1]]


def mix_row(row_list):
    rst = []
    for r in row_list:
        rst = rst + list(r)
    rst = list(set(rst))
    rst = sorted(rst)
    return rst
