import scipy.sparse
import scipy.io

data = scipy.io.loadmat("../data/cora_src.mat")

A = scipy.sparse.csr_matrix(data['A'])  # scipy.sparse.csr.csr_matrix
X0 = scipy.sparse.csr_matrix(data['X0'])  # scipy.sparse.csr.csr_matrix
W0 = data['W0']  # numpy.ndarray
B0 = data['B0']  # numpy.ndarray
D0 = scipy.sparse.csr_matrix(data['D0'])  # scipy.sparse.csr.csr_matrix
X1 = scipy.sparse.csr_matrix(data['X1'])  # scipy.sparse.csr.csr_matrix
W1 = data['W1']  # numpy.ndarray
B1 = data['B1']  # numpy.ndarray
D1 = scipy.sparse.csr_matrix(data['D1'])  # scipy.sparse.csr.csr_matrix
X2 = scipy.sparse.csr_matrix(data['X2'])  # scipy.sparse.csr.csr_matrix


def get_row(M: scipy.sparse.csr_matrix, rowId):
    return M.indices[M.indptr[rowId]:M.indptr[rowId + 1]]


M, N = A.shape
K, C = W0.shape

print(W0.shape, W1.shape)





