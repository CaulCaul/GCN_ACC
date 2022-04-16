from functional import *


class BufferPeace:
    def __init__(self, matrix_name: str, row_range: tuple, col_range: tuple, density: float = 1.0):
        self.name = matrix_name
        self.range = (row_range, col_range)  # ((r0, r1), (c0, c1))
        self.size = (row_range[1] - row_range[0]) * (col_range[1] - col_range[0]) * density


class Buffer:
    def __init__(self, buffer_size: int = 1000_000_000):
        self.size = buffer_size
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

    def load_elements(self, elements: BufferPeace):
        self.check_matrix(elements.name)
        self.bufferPeaces[elements.name].add(elements.range)
        self.increase_used(elements.size)
        return elements.size

    def evict_elements(self, elements: BufferPeace):
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

    def add_matrix(self, M, name: str):
        self.dict[name] = len(self.datalist)
        self.datalist.append(M)

    def select_elements(self, matrix_name: str, row_range: tuple = None, col_range: tuple = None):
        M = self.datalist[self.dict[matrix_name]]
        r, c = M.shape
        RowRange = row_range
        ColRange = col_range
        if row_range == None:
            RowRange = (0, r)
        if col_range == None:
            ColRange = (0, c)
        return BufferPeace(matrix_name, RowRange, ColRange, get_density(M))


class Accelerator:
    def __init__(self, num_MAC: int):
        self.num_MAC = num_MAC
        self.dataset = Dataset()
        self.buffer = Buffer()

        self.MUL_counter = 0
        self.DRAM_acc_counter = 0

