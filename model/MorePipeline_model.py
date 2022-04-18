
import sys, os
sys.path.append(os.getcwd())
from functional.Accelerator import Accelerator
import dgl

g = (dgl.data.CoraGraphDataset())[0]
AdjacentMatrix = g.adj(scipy_fmt='csr')



MP = Accelerator(16)

