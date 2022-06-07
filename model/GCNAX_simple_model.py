import math
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_adjacent(g):
    num_N = g.num_nodes()
    num_E = g.num_edges()
    A = torch.zeros(num_N, num_N, dtype=torch.bool)
    for i in range(num_E):
        A[g.edges()[0][i]][g.edges()[1][i]] = True
    return A


def GCNAX_counter(g, tiling_para_L1, tiling_para_L2, num_MAC=16):
    num_N, num_F = g.ndata['feat'].size()
    A = g.adj(scipy_fmt='csr')

    M, N, K, C = num_N, num_N, num_F, num_F
    Tn0, Tc0, Tk, Tn1, Tc1, Tm = tiling_para_L1

    dens_A = len(A.indices) / (num_N * num_N)
    dens_X0 = 0.0127
    dens_X1 = 0.78

    # dens_A  = 1
    # dens_X0 = 1
    # dens_X1 = 1

    counter_MUL_cyc = 0
    counter_DRAM_acc = 0

    # layer 1
    for n0 in range(0, N, Tn0):
        for c0 in range(0, C, Tc0):
            for k in range(0, K, Tk):
                Dtn0 = min(n0 + Tn0, N) - n0
                Dtc0 = min(c0 + Tc0, C) - c0
                Dtk = min(k + Tk, K) - k

                # load elements of X
                # load elements of W
                # (load elements of B)
                counter_DRAM_acc += dens_X0 * Dtn0 * Dtk
                counter_DRAM_acc += Dtk * Dtc0

                # B[tn0][tc0]+=X[tn0][tk]*W[tk][tc0]
                counter_MUL_cyc += dens_X0 * (Dtn0 * Dtc0 * Dtk)

                # (store elements of B)
                # counter_DRAM_acc += (dens_X0 * Dtn0 * Dtk) * Dtc0

            for m in range(0, M, Tm):
                Dtm = min(m + Tm, M) - m
                Dtc1 = min(c0 + Tc0, C) - c0
                Dtn1 = min(n0 + Tn0, N) - n0

                # (load elements of B)
                # load elements of A
                # load elements of O
                counter_DRAM_acc += dens_A * Dtm * Dtn1
                counter_DRAM_acc += Dtm * Dtc1

                # O[tm][tc1]+=A[tm][tn1]*B[tn1][tc1]
                counter_MUL_cyc += dens_A * (Dtm * Dtc1 * Dtn1)

                # store elements of O
                counter_DRAM_acc += Dtm * Dtc1

    # layer 2

    Tn0, Tc0, Tk, Tn1, Tc1, Tm = tiling_para_L2

    for n0 in range(0, N, Tn0):
        for c0 in range(0, C, Tc0):
            for k in range(0, K, Tk):
                Dtn0 = min(n0 + Tn0, N) - n0
                Dtc0 = min(c0 + Tc0, C) - c0
                Dtk = min(k + Tk, K) - k

                # load elements of X
                # load elements of W
                # (load elements of B)
                counter_DRAM_acc += dens_X1 * Dtn0 * Dtk
                counter_DRAM_acc += Dtk * Dtc0

                # B[tn0][tc0]+=X[tn0][tk]*W[tk][tc0]
                counter_MUL_cyc += dens_X1 * (Dtn0 * Dtc0 * Dtk)

                # (store elements of B)
                # counter_DRAM_acc += (dens_X1 * Dtn0 * Dtk) * Dtc0

            for m in range(0, M, Tm):
                Dtm = min(m + Tm, M) - m
                Dtc1 = min(c0 + Tc0, C) - c0
                Dtn1 = min(n0 + Tn0, N) - n0

                # (load elements of B)
                # load elements of A
                # load elements of O
                counter_DRAM_acc += dens_A * Dtm * Dtn1
                counter_DRAM_acc += Dtm * Dtc1

                # O[tm][tc1]+=A[tm][tn1]*B[tn1][tc1]
                counter_MUL_cyc += dens_A * (Dtm * Dtc1 * Dtn1)

                # store elements of O
                counter_DRAM_acc += Dtm * Dtc1

    counter_MUL_cyc /= num_MAC

    print("\nGCNAX at layers = 2 , MACs =", num_MAC, ", standard on chip cache.")
    print("MUL cycles  = ", counter_MUL_cyc)
    print("DRAM access = ", counter_DRAM_acc, "(about", (counter_DRAM_acc * 8) // (1024 * 1024), "MB)")

    return counter_MUL_cyc, counter_DRAM_acc


def GCNAX_formulaic_counter(g, dens_X, tiling_para_L1, tiling_para_L2, num_MAC=16, fusion: bool = True):
    num_N, num_F = g.ndata['feat'].size()
    A = g.adj(scipy_fmt='csr')

    M, N, K, C = num_N, num_N, num_F, num_F

    dens_A = len(A.indices) / (num_N * num_N)  # 这一步算的是A本身的稀疏度，实际上应该算normalized A的，不过误差不大
    # print(dens_A)
    dens_X0, dens_X1 = dens_X

    # dens_A  = 1
    # dens_X0 = 1
    # dens_X1 = 1

    counter_DRAM_acc = 0

    # layer 1
    Tn0, Tc0, Tk, Tn1, Tc1, Tm = tiling_para_L1

    aX = float(N * C * K) / (Tn0 * Tc0 * Tk)
    aW = float(N * C * K) / (Tn0 * Tc0 * Tk)

    if fusion:
        aB1 = 0
        aB2 = 0
        aA = float(M * C * N) / (Tm * Tc0 * Tn0)
        aO = 2 * float(M * C * N) / (Tm * Tc0 * Tn0)
    else:
        aB1 = float(N * C) / (Tn0 * Tc0)
        aB2 = float(M * C * N) / (Tm * Tc1 * Tn1)
        aA = float(M * C * N) / (Tm * Tc1 * Tn1)
        aO = float(M * C) / (Tm * Tc1)

    SX = dens_X0 * Tn0 * Tk
    SW = Tk * Tc0
    SB1 = Tn0 * Tc0
    SB2 = Tn1 * Tc1
    SA = dens_A * Tm * Tn1
    SO = Tm * Tc1

    counter_DRAM_acc += aX * SX + aW * SW + aB1 * SB1 + aB2 * SB2 + aA * SA + aO * SO

    # layer 2
    Tn0, Tc0, Tk, Tn1, Tc1, Tm = tiling_para_L2

    aX = float(N * C * K) / (Tn0 * Tc0 * Tk)
    aW = float(N * C * K) / (Tn0 * Tc0 * Tk)

    if fusion:
        aB1 = 0
        aB2 = 0
        aA = float(M * C * N) / (Tm * Tc0 * Tn0)
        aO = 2 * float(M * C * N) / (Tm * Tc0 * Tn0)
    else:
        aB1 = float(N * C) / (Tn0 * Tc0)
        aB2 = float(M * C * N) / (Tm * Tc1 * Tn1)
        aA = float(M * C * N) / (Tm * Tc1 * Tn1)
        aO = float(M * C) / (Tm * Tc1)

    SX = dens_X1 * Tn0 * Tk
    SW = Tk * Tc0
    SB1 = Tn0 * Tc0
    SB2 = Tn1 * Tc1
    SA = dens_A * Tm * Tn1
    SO = Tm * Tc1
    counter_DRAM_acc += aX * SX + aW * SW + aB1 * SB1 + aB2 * SB2 + aA * SA + aO * SO

    # print("\nGCNAX at layers = 2 , MACs =", num_MAC, ", standard on chip cache.")
    print("DRAM access = ", counter_DRAM_acc, "(about", (counter_DRAM_acc * 8) // (1024 * 1024), "MB)")

    return counter_DRAM_acc
