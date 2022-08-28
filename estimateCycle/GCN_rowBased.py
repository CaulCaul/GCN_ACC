import scipy.io
import scipy.sparse

from util.functions import *

load_latency = 50


def get_num_cycle_rowbase(dataset: str, mac_num: int = 16) -> (int, int):
    data = scipy.io.loadmat("../data/" + dataset.lower() + "_src.mat")

    A = scipy.sparse.csr_matrix(data['A'])  # scipy.sparse.csr.csr_matrix
    X0 = scipy.sparse.csr_matrix(data['X0'])  # scipy.sparse.csr.csr_matrix
    W0 = data['W0']  # numpy.ndarray
    B0 = data['B0']  # numpy.ndarray
    D0 = scipy.sparse.csr_matrix(data['D0'])  # scipy.sparse.csr.csr_matrix
    X1 = scipy.sparse.csr_matrix(data['X1'])   # scipy.sparse.csr.csr_matrix
    W1 = data['W1']  # numpy.ndarray
    B1 = data['B1']  # numpy.ndarray
    D1 = scipy.sparse.csr_matrix(data['D1'])  # scipy.sparse.csr.csr_matrix
    X2 = scipy.sparse.csr_matrix(data['X2'])  # scipy.sparse.csr.csr_matrix

    M, N = A.shape
    K, C = W0.shape

    mul_cycles, load_cycles = 0, 0
    m, l = compute_cycles(A, X0, mac_num)
    mul_cycles += m
    load_cycles += l
    m, l = compute_cycles(D0, W0, mac_num)
    mul_cycles += m
    load_cycles += l
    m, l = compute_cycles(A, X1, mac_num)
    mul_cycles += m
    load_cycles += l
    m, l = compute_cycles(D1, W1, mac_num)
    mul_cycles += m
    load_cycles += l
    return mul_cycles, load_cycles


def compute_cycles(A: scipy.sparse.csr.csr_matrix, B: scipy.sparse.csr.csr_matrix, mac_num: int) -> int:
    M, N = A.shape
    mul_cycles, load_cycles = 0, 0

    for ar in range(0, M, mac_num):
        ar1 = min(ar + mac_num, M)
        load_cycles += load_latency * 2  # load A, then load B

        tmp_cycles = 0
        for i in range(ar, ar1):
            row = get_row(A, i)
            for j in row:
                tmp_cycles += len(get_row(B, j))

        mul_cycles += tmp_cycles // mac_num

        load_cycles += load_latency

    return mul_cycles, load_cycles

