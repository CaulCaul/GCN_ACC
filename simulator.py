from model.GCNAX_simple_model import *
from model.MorePipeline_simple_model import *

# g = dgl.graph(([0, 0, 1, 1, 2, 3, 3, 4],
#                [1, 3, 2, 4, 0, 1, 4, 2]))

# dataset = dgl.data.CoraGraphDataset()
# dataset = dgl.data.PubmedGraphDataset()

# A = g.adj(scipy_fmt='csr')

# print(A)
# print(A.dtype)
# print(len(A.data))
# print(len(A.indices))
# print(len(A.indptr))
#
# print(A.shape)
#
# print(len(A.indices) / (A.shape[0] * A.shape[1]))


# MP_mul, MP_dram = More_Pipeline_model_counter(g)
#
# print("\nRatio : ", MP_mul/GCNAX_mul, ",", MP_dram/GCNAX_dram)

univers_fusion_para = (2048, 16, 16, 2048, 16, 16)
cora_best_para_L1 = (2708,16,1,2708,16,1)
cora_best_para_L2 = (2708,7,1,2708,7,1)


g = dgl.data.CoraGraphDataset()[0]
GCNAX_counter(g, univers_fusion_para, univers_fusion_para)
GCNAX_counter(g, cora_best_para_L1, cora_best_para_L2)

