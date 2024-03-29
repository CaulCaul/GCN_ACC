import scipy.sparse
import scipy.io

from util import functions

data = scipy.io.loadmat("../data/pubmed_src.mat")
# data = scipy.io.loadmat("../data/pubmed_src.mat")

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


def get_row(M, rowId):
    if type(M) == scipy.sparse.csr_matrix:
        return M.indices[M.indptr[rowId]:M.indptr[rowId + 1]]
    else:
        return M[rowId]


M, N = A.shape
# K, C = W0.shape

tM = 128
load_latency = 50
mac_num = 16

max_cache = 0
Dram_counter = 0
mul_cycle_counter = 0
load_cycle_counter = 0

tmp_cycle = 0

max_part = 50 * 16 * 2

for dM in range(0, M, tM):
    print(dM, "/", M, end="\r")

    dM1 = min(dM + tM, M)
    Ar = []
    cacheA0 = 0
    for i in range(dM, dM1):  # load A row
        row = get_row(A, i)
        cacheA0 += len(row)
        Dram_counter += len(row)
        Ar.append(row)
    load_cycle_counter += load_latency

    cacheX0 = 0
    for i in functions.mix_row(Ar):  # load X0 row
        row = get_row(X0, i)
        cacheX0 += len(row)
        Dram_counter += len(row)
    load_cycle_counter += load_latency

    D0r = []
    cacheD0 = 0
    for i in range(dM, dM1):  # compute D0 row
        row = get_row(D0, i)
        cacheD0 += len(row)
        D0r.append(row)

    multiplication_times = 0
    for r in Ar:
        for c in r:
            multiplication_times += len(get_row(X0, c))
    mul_cycle_counter += multiplication_times / mac_num
    tmp_cycle = multiplication_times / mac_num

    cacheW0 = 0
    for i in functions.mix_row(D0r):  # load W0 row
        row = W0[i]
        cacheW0 += len(row)  # 2
        Dram_counter += len(row)
    load_cycle_counter += max(load_latency - tmp_cycle, 0)  # prefetching

    X1r = []
    cacheX1 = 0
    for i in range(dM, dM1):  # compute X1 row
        row = get_row(X1, i)
        cacheX1 += len(row)
        X1r.append(row)

    multiplication_times = 0
    for r in D0r:
        for c in r:
            multiplication_times += len(get_row(W0, c))
    mul_cycle_counter += multiplication_times / mac_num
    tmp_cycle = multiplication_times / mac_num

    cacheW1 = 0
    for i in functions.mix_row(X1r):  # load W1 row
        row = W1[i]
        cacheW1 += len(row)  # 1
        Dram_counter += len(row)
    load_cycle_counter += max(load_latency - tmp_cycle, 0)

    cacheB1 = 0
    for i in range(dM, dM1):  # compute B1
        row = B1[i]
        cacheB1 += len(row)

        Dram_counter += len(row)  # store B1

    multiplication_times = 0
    for r in X1r:
        for c in r:
            multiplication_times += len(get_row(W1, c))
    mul_cycle_counter += multiplication_times / mac_num

    # store B1
    load_cycle_counter += load_latency

    # # X2 = A * B1
    # Dram_counter += len(A.indices)  # load A
    # Dram_counter += 2 * N * K  # load X2 & store X2

    cacheX0 = min(cacheX0, max_part)
    cacheW0 = min(cacheW0, max_part)
    cacheW1 = min(cacheW1, max_part)
    max_cache = max(max_cache,
                    cacheA0 + cacheX0 + cacheD0,
                    cacheD0 + cacheW0 + cacheX1,
                    cacheX1 + cacheW1 + cacheB1)

for dM in range(0, M, tM):
    dM1 = min(dM + tM, M)
    Ar = []
    cacheA0 = 0
    for i in range(dM, dM1):  # load A row
        row = get_row(A, i)
        cacheA0 += len(row)
        Dram_counter += len(row)
        Ar.append(row)
    load_cycle_counter += load_latency

    cacheB1 = 0
    for i in functions.mix_row(Ar):  # load B1 row
        row = B1[i]
        cacheB1 += len(row)  # 3
        Dram_counter += len(row)
    load_cycle_counter += load_latency

    cacheX2 = 0
    for i in range(dM, dM1):  # compute X2 row
        row = get_row(D0, i)
        cacheX2 += len(row)
        Dram_counter += len(row)  # store X2

    multiplication_times = 0
    for r in Ar:
        for c in r:
            multiplication_times += len(get_row(B1, c))
    mul_cycle_counter += multiplication_times / mac_num

    cacheB1 = min(cacheB1, max_part)
    max_cache = max(max_cache, cacheA0 + cacheB1 + cacheX2)

# 减少cache的大小


print("\n")
print("tM :", tM)
print("Max cache :", max_cache, "(about", "{:.2f}".format((max_cache * 8) / 1024), "KB)")
print("Dram access :", Dram_counter, "(about", "{:.2f}".format((Dram_counter * 8) / (1024 * 1024)), "MB)")
print("Cycles :", mul_cycle_counter, load_cycle_counter)
