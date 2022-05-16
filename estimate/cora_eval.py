import dgl
import pandas as pd
import scipy.sparse
import scipy.io

import functional

data = scipy.io.loadmat("cora_src.mat")

A = scipy.sparse.csr_matrix(data['A'])  # scipy.sparse.csr.csr_matrix
X0 = scipy.sparse.csr_matrix(data['X0'])  # scipy.sparse.csr.csr_matrix
W0 = data['W0']  # numpy.ndarray
B0 = data['B0']  # numpy.ndarray
D0 = scipy.sparse.csr_matrix(data['D0'])  # scipy.sparse.csr.csr_matrix
X1 = scipy.sparse.csr_matrix(data['X1'])  # scipy.sparse.csr.csr_matrix
W1 = data['W1']  # numpy.ndarray
B1 = data['B1']  # numpy.ndarray
D1 = scipy.sparse.csr_matrix(data['D1'])  # scipy.sparse.csr.csr_matrix
X2 = scipy.sparse.csr_matrix(data['X2'])  # scipy.sparse.csr.csr_matrix


def get_row(M: scipy.sparse.csr_matrix, rowId):
    return M.indices[M.indptr[rowId]:M.indptr[rowId + 1]]


M, N = A.shape
K, C = W0.shape

tM = 3

cycle = 0
cycle += 50

max_cache = 0
Dram_counter = 0

for dM in range(0, M, tM):
    print("\r", dM, "/", M, end="")

    dM1 = min(dM + tM, M)
    Ar = []
    cacheA = 0
    for i in range(dM, dM1):  # load A row
        row = get_row(A, i)
        cacheA += len(row)
        Dram_counter += len(row)
        Ar.append(row)

    cacheX0 = 0
    for i in functional.mix_row(Ar):  # load X0 row
        row = get_row(X0, i)
        cacheX0 += len(row)
        Dram_counter += len(row)

    D0r = []
    cacheD0 = 0
    for i in range(dM, dM1):  # compute D0 row
        row = get_row(D0, i)
        cacheD0 += len(row)
        D0r.append(row)

    cacheW0 = 0
    for i in functional.mix_row(D0r):  # load W0 row
        row = W0[i]
        cacheW0 += len(row)
        Dram_counter += len(row)

    X1r = []
    cacheX1 = 0
    for i in range(dM, dM1):  # compute X1 row
        row = get_row(X1, i)
        cacheX1 += len(row)
        X1r.append(row)

    cacheW1 = 0
    for i in functional.mix_row(X1r):  # load W1 row
        row = W1[i]
        cacheW1 += len(row)
        Dram_counter += len(row)

    cacheB1 = 0
    for i in range(dM, dM1):  # compute B1
        row = B1[i]
        cacheB1 += len(row)

    max_cache = max(max_cache,
                    cacheA + cacheX0 + cacheD0,
                    cacheD0 + cacheD0 + cacheX1,
                    cacheX1 + cacheW1 + cacheB1)

print("\n")
print("tM :", tM)
print("Max cache :", max_cache)
print("Dram access :", Dram_counter)
