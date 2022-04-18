
from model.GCNAX_simple_model import *
from model.MorePipeline_simple_model import *

# g = dgl.graph(([0, 0, 1, 1, 2, 3, 3, 4],
#                [1, 3, 2, 4, 0, 1, 4, 2]))

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

A = g.adj(scipy_fmt='csr')

# print(A)
# print(A.dtype)
# print(len(A.data))
# print(len(A.indices))
# print(len(A.indptr))
#
# print(A.shape)
#
# print(len(A.indices) / (A.shape[0] * A.shape[1]))



GCNAX_mul, GCNAX_dram = GCNAX_counter(g)
MP_mul, MP_dram = More_Pipeline_model_counter(g)

print("\nRatio : ", MP_mul/GCNAX_mul, ",", MP_dram/GCNAX_dram)





