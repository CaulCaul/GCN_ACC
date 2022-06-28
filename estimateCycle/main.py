
import GCN_innerProduct
import GCN_rowBased

print("Inner:")
print("Cora:", GCN_innerProduct.get_num_cycle_inner("cora"))
print("Pubmed:", GCN_innerProduct.get_num_cycle_inner("pubmed"))
print("Row:")
print("Cora:", GCN_rowBased.get_num_cycle_rowbase("cora"))
print("Pubmed:", GCN_rowBased.get_num_cycle_rowbase("pubmed"))

