import math
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def More_Pipeline_model_counter(g, num_MAC=16):

    num_N, num_F = g.ndata['feat'].size()
    A = g.adj(scipy_fmt='csr')

    dens_A = len(A.indices) / (num_N * num_N)
    dens_X1 = 0.0127
    dens_X2 = 0.78

    dens_AX1 = 0.06

    counter_MUL_cyc = 0
    counter_DRAM_acc = 0

    for ar in range(0, num_N):
        # load row of A
        counter_DRAM_acc += dens_A * num_N
        for xc in range(0, num_F):
            # load column of X1
            counter_DRAM_acc += dens_A * num_N
            counter_MUL_cyc += num_N * dens_A
        # get row of AX1

        for w1c in range(0, num_F):
            # load column of W1
            counter_DRAM_acc += dens_AX1 * num_F
            counter_MUL_cyc += dens_AX1 * num_F
        # get row of X2 = ReLU(AX1W1)

        for w2c in range(0, num_F):
            # load column of W2
            counter_DRAM_acc += dens_X2 * num_F
            counter_MUL_cyc += dens_X2 * num_F
        # get row of X2W2
        # store X2W2
        counter_DRAM_acc += num_F


    # 最后一步乘法如何计算？
    for ar in range(0, num_N):
        # load row of A
        counter_DRAM_acc += dens_A * num_N
        for x2w2c in range(0, num_F):
            # load column of X2W2
            counter_DRAM_acc += dens_A * num_N
            counter_MUL_cyc += num_N * dens_A

        counter_DRAM_acc += num_N

    counter_MUL_cyc /= num_MAC




    print("\nMore Pipeline model at layers = 2 , MACs =", num_MAC, "small on chip cache.")
    print("MUL cycles  = ", counter_MUL_cyc)
    print("DRAM access = ", counter_DRAM_acc)

    return counter_MUL_cyc, counter_DRAM_acc