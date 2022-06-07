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
univers_para = (2048, 16, 16, 16, 16, 2048)

cora_best_para_L1 = (2708, 16, 1, 2708, 16, 1)
cora_best_para_L2 = (2708, 7, 1, 2708, 7, 1)

pubmed_best_para_L1 = (3073, 16, 1, 1, 16, 3073)
pubmed_best_para_L2 = (3000, 3, 1, 1025, 3, 3000)

cora_dens_X = (0.0127, 0.78)
pubmed_dens_X = (0.1, 0.776)

print("Cora")
g = dgl.data.CoraGraphDataset()[0]
# GCNAX_counter(g, univers_fusion_para, univers_fusion_para)
# GCNAX_counter(g, cora_best_para_L1, cora_best_para_L2)
GCNAX_formulaic_counter(g, cora_dens_X, univers_fusion_para, univers_fusion_para)
GCNAX_formulaic_counter(g, cora_dens_X, cora_best_para_L1, cora_best_para_L2)

print("Pubmed")
g = dgl.data.PubmedGraphDataset()[0]
GCNAX_formulaic_counter(g, pubmed_dens_X, univers_fusion_para, univers_fusion_para)
GCNAX_formulaic_counter(g, pubmed_dens_X, univers_para, univers_para, fusion=False)
GCNAX_formulaic_counter(g, pubmed_dens_X, pubmed_best_para_L1, pubmed_best_para_L2, fusion=False)
