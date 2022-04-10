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

def GCNAX_counter(g, num_MAC = 16):

    num_N, num_F = g.ndata['feat'].size()
    A = g.adj(scipy_fmt='csr')

    M, N, K, C = num_N, num_N, num_F, num_F
    Tn0, Tc0, Tk, Tn1, Tc1, Tm = 2048, 16, 16, 2048, 16, 16

    dens_A = len(A.indices) / (num_N * num_N)
    dens_X1 = 0.0127
    dens_X2 = 0.78

    # dens_A  = 1
    # dens_X1 = 1
    # dens_X2 = 1

    counter_MUL_cyc = 0
    counter_DRAM_acc = 0

    # layer 1
    for n0 in range(0, N, Tn0):
        for c0 in range(0, C, Tc0):
            for k in range(0, K, Tk):
                Dtn0 = min(n0+Tn0, N) - n0
                Dtc0 = min(c0+Tc0, C) - c0
                Dtk  = min(k+Tk, K) - k

                # load elements of X
                # load elements of W
                # load elements of B
                counter_DRAM_acc += (dens_X1 * Dtn0 * Dtk) * (1 + Dtc0 + Dtc0)

                # B[tn0][tc0]+=X[tn0][tk]*W[tk][tc0]
                counter_MUL_cyc += dens_X1 * (Dtn0 * Dtc0 * Dtk)

                counter_DRAM_acc += (dens_X1 * Dtn0 * Dtk) * Dtc0

            # (store elements of B)

            for m in range(0, M, Tm):
                Dtm  = min(m+Tm, M) - m
                Dtc1 = min(c0+Tc0, C) - c0
                Dtn1 = min(n0+Tn0, N) - n0

                # (load elements of B)
                # load elements of A
                # load elements of O
                counter_DRAM_acc += (dens_A * Dtm * Dtn1) * (1 + Dtc1 + Dtc1)

                # O[tm][tc1]+=A[tm][tn1]*B[tn1][tc1]
                counter_MUL_cyc += dens_A * (Dtm * Dtc1 * Dtn1)

                counter_DRAM_acc += (dens_A * Dtm * Dtn1) * Dtc1

                # store elements of O

    # layer 2
    for n0 in range(0, N, Tn0):
        for c0 in range(0, C, Tc0):
            for k in range(0, K, Tk):
                Dtn0 = min(n0+Tn0, N) - n0
                Dtc0 = min(c0+Tc0, C) - c0
                Dtk  = min(k+Tk, K) - k

                # load elements of X
                # load elements of W
                # load elements of B
                counter_DRAM_acc += (dens_X2 * Dtn0 * Dtk) * (1 + Dtc0 + Dtc0)

                # B[tn0][tc0]+=X[tn0][tk]*W[tk][tc0]
                counter_MUL_cyc += dens_X2 * (Dtn0 * Dtc0 * Dtk)

                counter_DRAM_acc += (dens_X2 * Dtn0 * Dtk) * Dtc0

            # (store elements of B)

            for m in range(0, M, Tm):
                Dtm  = min(m+Tm, M) - m
                Dtc1 = min(c0+Tc0, C) - c0
                Dtn1 = min(n0+Tn0, N) - n0

                # (load elements of B)
                # load elements of A
                # load elements of O
                counter_DRAM_acc += (dens_A * Dtm * Dtn1) * (1 + Dtc1 + Dtc1)

                # O[tm][tc1]+=A[tm][tn1]*B[tn1][tc1]
                counter_MUL_cyc += dens_A * (Dtm * Dtc1 * Dtn1)

                counter_DRAM_acc += (dens_A * Dtm * Dtn1) * Dtc1

                # store elements of O

    counter_MUL_cyc /= num_MAC

    print("\nGCNAX at layers = 2 , MACs =", num_MAC, ", No on chip cache.")
    print("MUL cycles  = ", counter_MUL_cyc)
    print("DRAM access = ", counter_DRAM_acc)

    return counter_MUL_cyc, counter_DRAM_acc









