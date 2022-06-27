import warnings

import scipy.sparse.csr
import torch


def get_density(M):
    r, c = M.shape
    return len(M.indices) / (c * r)


class MatrixProfile:
    def __init__(self, source_matrix):
        if type(source_matrix) == scipy.sparse.csr.csr_matrix:
            self.shape = source_matrix.shape
            self.density = len(source_matrix.indices) / (self.shape[0] * self.shape[1])
        elif type(source_matrix) == torch.Tensor:
            r, c = source_matrix.shape
            self.shape = (r, c)
            self.density = 1.0
        elif type(source_matrix) == tuple:
            self.shape = (source_matrix[0], source_matrix[1])
            self.density = source_matrix[2]
        else:
            raise RuntimeError("MatrixProfile don't support this type.")


class BufferPiece:
    def __init__(self, matrix_name: str, row_range: (int, int), col_range: (int, int), density: float = 1.0):
        self.name = matrix_name
        self.density = density
        self.range = (row_range, col_range)  # ((r0, r1), (c0, c1))
        self.size = (row_range[1] - row_range[0]) * (col_range[1] - col_range[0]) * density
        self.shape = (row_range[1] - row_range[0], col_range[1] - col_range[0])


class Buffer:
    def __init__(self, buffer_size: int = 1000_000_000):
        self.size = buffer_size
        self.max_used = 0
        self.used = 0
        self.matrix_set = set()
        self.bufferPeaces = {}

    def reset_state(self):
        self.max_used = 0
        self.used = 0
        self.matrix_set = set()
        self.bufferPeaces = {}

    def check_matrix(self, matrix_name: str):
        if matrix_name not in self.matrix_set:
            self.matrix_set.add(matrix_name)
            self.bufferPeaces[matrix_name] = set()

    def increase_used(self, data_size: int):
        if data_size + self.used > self.size:
            raise RuntimeError("Buffer overflow.")
        self.used += data_size
        if self.max_used < self.used:
            self.max_used = self.used

    def load_elements(self, elements: BufferPiece):
        self.check_matrix(elements.name)
        self.bufferPeaces[elements.name].add(elements.range)
        self.increase_used(elements.size)
        return elements.size

    def evict_elements(self, elements: BufferPiece):
        self.check_matrix(elements.name)
        if elements.range in self.bufferPeaces[elements.name]:
            self.bufferPeaces[elements.name].remove(elements.range)
            self.used -= elements.size
            return elements.size
        else:
            raise RuntimeError("Remove non-existent elements from buffer.")


class Dataset:
    def __init__(self):
        self.dict = {}
        self.datalist = []

    # 将矩阵添加至列表中，并为其指定一个唯一的字符串索引
    def add_matrix(self, M, name: str):
        if name in self.dict:
            warnings.warn("The matrix name '"+name+"' is exist.")
        else:
            self.dict[name] = len(self.datalist)
            self.datalist.append(MatrixProfile(M))

    # 选定某个矩阵的一块，返回BufferPiece
    def select_elements(self, matrix_name: str, row_range: (int, int) = None, col_range: (int, int) = None):
        M = self.datalist[self.dict[matrix_name]]
        r, c = M.shape
        RowRange = row_range
        ColRange = col_range
        if row_range is None:
            RowRange = (0, r)
        if col_range is None:
            ColRange = (0, c)
        return BufferPiece(matrix_name, RowRange, ColRange, M.density)

    def get_density(self, matrix_name: str):
        return self.datalist[self.dict[matrix_name]].density
