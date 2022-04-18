from functional.functional import *


class BufferPiece:
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
        self.dict[name] = len(self.datalist)
        self.datalist.append(M)

    # 选定某个矩阵的一块，返回BufferPiece
    def select_elements(self, matrix_name: str, row_range:(int, int) = None, col_range: (int, int) = None):
        M = self.datalist[self.dict[matrix_name]]
        r, c = M.shape
        RowRange = row_range
        ColRange = col_range
        if row_range is None:
            RowRange = (0, r)
        if col_range is None:
            ColRange = (0, c)
        return BufferPiece(matrix_name, RowRange, ColRange, get_density(M))


class Accelerator:
    def __init__(self, num_MAC: int):
        self.num_MAC = num_MAC
        self.dataset = Dataset()
        self.buffer = Buffer()

        self.MUL_counter = 0
        self.DRAM_acc_counter = 0

    def reset_state(self):
        self.MUL_counter = 0
        self.DRAM_acc_counter = 0
        self.buffer.reset_state()

    def print_state(self):
        print("Current State:")
        print("MAC number :", self.num_MAC)
        print("Multiplication counter :", self.MUL_counter)
        print("Dram access counter :", self.DRAM_acc_counter)
        print("Maximum cache usage :", self.buffer.max_used)

    # 将矩阵添加至列表中，并为其指定一个唯一的字符串索引
    def add_matrix(self, M, name: str):
        self.dataset.add_matrix(M, name)

    def load_block(self, name: str, row_range: (int, int) = None, col_range: (int, int) = None):
        piece = self.dataset.select_elements(name, row_range=row_range, col_range=col_range)
        self.DRAM_acc_counter += self.buffer.load_elements(piece)

    def load_row(self, name: str, rowId: int):
        self.load_block(name, row_range=(rowId, rowId+1))

    def load_col(self, name: str, colId: int):
        self.load_block(name, col_range=(colId, colId+1))

    def evict_block(self, name: str, row_range: (int, int) = None, col_range: (int, int) = None, store: bool = False):
        piece = self.dataset.select_elements(name, row_range=row_range, col_range=col_range)
        datasize = self.buffer.evict_elements(piece)
        if store:
            self.DRAM_acc_counter += datasize

    def evict_row(self, name: str, rowId: int, store: bool = False):
        self.evict_block(name, row_range=(rowId, rowId + 1), store=store)

    def evict_col(self, name: str, colId: int, store: bool = False):
        self.evict_block(name, col_range=(colId, colId + 1), store=store)


    # 还需实现对于矩阵乘法支持的相应成员函数


























