import dgl
import numpy as np
import scipy.io, scipy.sparse


def get_density(M):
    r, c = M.shape
    return len(M.indices) / (c * r)


def gen_data(g, dataset_name: str):
    print("Start gen_data :", dataset_name)
    A = g.adj(scipy_fmt="csr")
    X0 = scipy.sparse.csr_matrix(g.ndata['feat'])
    n, k = X0.shape
    W0 = np.random.rand(k, k)
    W0 = 2 * W0 - 1
    # print(type(W0))
    B0 = X0.dot(W0)
    # print(type(B0), B0.shape)
    D0 = A.dot(X0)

    X1 = D0.dot(W0)
    X1 = 1 * (X1 > 0) * X1
    X1 = scipy.sparse.csr_matrix(X1)
    print(type(X1), X1.shape, get_density(X1))

    W1 = np.random.rand(k, k)
    W1 = 2 * W1 - 1
    B1 = X1 * W1
    D1 = A * X1

    X2 = A * X1 * W1
    X2 = 1 * (X2 > 0) * X2
    X2 = scipy.sparse.csr_matrix(X2)
    # print(type(X1), X1.shape, get_density(X2))

    scipy.io.savemat(dataset_name + '_src.mat', {'A': A,
                                                 'X0': X0,
                                                 'W0': W0,
                                                 'B0': B0,
                                                 'D0': D0,
                                                 'X1': X1,
                                                 'W1': W1,
                                                 'B1': B1,
                                                 'D1': D1,
                                                 'X2': X2})


gen_data(dgl.data.CoraGraphDataset()[0], "cora")
gen_data(dgl.data.PubmedGraphDataset()[0], "pubmed")

# gen_data(dgl.data.RedditDataset()[0], "reddit")  # too large
