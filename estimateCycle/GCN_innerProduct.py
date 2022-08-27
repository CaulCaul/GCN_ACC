import scipy.io
import scipy.sparse

from util.functions import *

load_latency = 50

# request_latency = 25
# data_latency -> DDR5


def get_num_cycle_inner(dataset: str, mac_num: int = 16) -> int:
    data = scipy.io.loadmat("../data/" + dataset.lower() + "_src.mat")

    A = scipy.sparse.csr_matrix(data['A'])  # scipy.sparse.csr.csr_matrix
    X0 = scipy.sparse.csc_matrix(data['X0'])  # scipy.sparse.csc.csc_matrix
    W0 = data['W0']  # numpy.ndarray
    B0 = data['B0']  # numpy.ndarray
    D0 = scipy.sparse.csr_matrix(data['D0'])  # scipy.sparse.csr.csr_matrix
    X1 = scipy.sparse.csc_matrix(data['X1'])  # scipy.sparse.csc.csc_matrix
    W1 = data['W1']  # numpy.ndarray
    B1 = data['B1']  # numpy.ndarray
    D1 = scipy.sparse.csr_matrix(data['D1'])  # scipy.sparse.csr.csr_matrix
    X2 = scipy.sparse.csr_matrix(data['X2'])  # scipy.sparse.csr.csr_matrix

    M, N = A.shape
    K0, C0 = W0.shape
    K1, C1 = W1.shape
    # print(M, N, K0, C0, K1, C1)

    cycles = 0
    load_cycles = 0

    # X0_col_wise = compute_cycles(X0)
    # W0_col_wise = compute_cycles(W0, row_wise=False)
    # X1_col_wise = compute_cycles(X1)
    # W1_col_wise = compute_cycles(W1, row_wise=False)

    for ar in range(0, M, mac_num):
        load_cycles += load_latency  # load A row
        cycles += compute_cycles(A, ar, X0)

    for dr in range(0, M, mac_num):
        load_cycles += load_latency  # load D0 row
        cycles += compute_cycles(D0, dr, W0, row_wise=False)

    for ar in range(0, M, mac_num):
        load_cycles += load_latency  # load A row
        cycles += compute_cycles(A, ar, X1)

    for dr in range(0, M, mac_num):
        load_cycles += load_latency  # load D0 row
        cycles += compute_cycles(D1, dr, W1, row_wise=False)

    print("({:d})".format(load_cycles))
    cycles += load_cycles
    return cycles


def compute_cycles(A, rowIdx, B, row_wise: bool = True, debug: bool = False) -> int:
    cycles = load_latency

    v_len = len(get_row(A, rowIdx))

    R, C = B.shape

    if type(B) == scipy.sparse.csc.csc_matrix:
        if debug: print("csc")
        for col in range(C - 1):
            l = len(get_col(B, col))
            cycles += max(min(l, v_len), load_latency)

        cycles += len(get_col(B, C - 1))
    elif type(B) == scipy.sparse.csr.csr_matrix:
        if debug: print("csr")
        for row in range(R - 1):
            l = len(get_row(B, row))
            cycles += max(min(l, v_len), load_latency)

        cycles += len(get_row(B, R - 1))
    elif type(B) == numpy.ndarray:
        if debug: print("dense")
        if not row_wise:
            B = B.transpose()
            R, C = B.shape

        for row in range(R - 1):
            l = len(B[row])
            cycles += max(min(l, v_len), load_latency)
        cycles += len(B[R - 1])

    return cycles
