from model.GCNAX_simple_model import *
from model.MorePipeline_simple_model import *

univers_fusion_para = (2048, 16, 16, 2048, 16, 16)
univers_para = (2048, 16, 16, 16, 16, 2048)

cora_best_para_L1 = (2708, 16, 1, 2708, 16, 1)
cora_best_para_L2 = (2708, 7, 1, 2708, 7, 1)

pubmed_best_para_L1 = (3073, 16, 1, 1, 16, 3073)
pubmed_best_para_L2 = (3000, 3, 1, 1025, 3, 3000)

cora_dens_X = (0.0127, 0.78)
pubmed_dens_X = (0.1, 0.776)

cora_layer_paras = (1433, 16, 7)
pubmed_layer_paras = (500, 16, 3)
g0 = dgl.data.CoraGraphDataset()[0]
g1 = dgl.data.PubmedGraphDataset()[0]

# print("More Pipeline")
# More_Pipeline_model_counter(g0, cora_layer_paras)
# More_Pipeline_model_counter(g1, pubmed_layer_paras)

print("Cora")
GCNAX_counter(g0, cora_layer_paras, univers_fusion_para, univers_fusion_para)
GCNAX_counter(g0, cora_layer_paras, cora_best_para_L1, cora_best_para_L2)
GCNAX_formulaic_counter(g0, cora_layer_paras, cora_dens_X, univers_fusion_para, univers_fusion_para)
GCNAX_formulaic_counter(g0, cora_layer_paras, cora_dens_X, cora_best_para_L1, cora_best_para_L2)

print("Pubmed")
GCNAX_formulaic_counter(g1, pubmed_layer_paras, pubmed_dens_X, univers_fusion_para, univers_fusion_para)
GCNAX_formulaic_counter(g1, pubmed_layer_paras, pubmed_dens_X, univers_para, univers_para, fusion=False)
GCNAX_formulaic_counter(g1, pubmed_layer_paras, pubmed_dens_X, pubmed_best_para_L1, pubmed_best_para_L2, fusion=False)

