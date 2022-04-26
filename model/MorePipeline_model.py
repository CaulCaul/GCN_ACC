import sys, os

sys.path.append(os.getcwd())
from functional.Accelerator import Accelerator
import dgl

g = (dgl.data.CoraGraphDataset())[0]
AdjacentMatrix = g.adj(scipy_fmt='csr')

num_N, num_F = g.ndata['feat'].size()

MP = Accelerator(16)

MP.add_matrix(AdjacentMatrix, "A")
MP.add_matrix((num_N, num_F, 0.0127), 'X1')
MP.add_matrix((num_N, num_F, 0.78), 'X2')
MP.add_matrix((num_F, num_F, 1.0), 'W1')
MP.add_matrix((num_F, num_F, 1.0), 'W2')

MP.add_matrix((num_N, num_F, 0.06), 'AX1')
MP.add_matrix((num_N, num_F, 1.0), 'X2W2')

for ar in range(0, num_N):
    print('\rCurrent row :', ar, '. Max cache usage :', MP.buffer.max_used, end='')
    Ar = MP.load_row('A', ar)
    for xc in range(0, num_F):  # 需要load整列吗？ && 优化计算速度
        X1c = MP.load_col('X1', xc)
        MP.matrix_mul(Ar, X1c, 'AX1')  # AX1[ar][xc]
        MP.evict_col('X1', xc)
    MP.evict_row('A', ar)

    AX1r = MP.select_row('AX1', ar)
    MP.generate_block(AX1r)

    for w1c in range(0, num_F):
        W1c = MP.load_col('W1', w1c)
        MP.matrix_mul(AX1r, W1c, 'X2')  # X2[ar][w1c]
        MP.evict_col('W1', w1c)
    MP.evict_row('AX1', ar)

    X2r = MP.select_row('X2', ar)
    MP.generate_block(X2r)

    MP.evict_row('X2', ar)






print("\n")

MP.print_state()


