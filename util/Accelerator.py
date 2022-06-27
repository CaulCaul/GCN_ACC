from functional.functional import *


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
        return piece

    def load_row(self, name: str, rowId: int):
        return self.load_block(name, row_range=(rowId, rowId + 1))

    def load_col(self, name: str, colId: int):
        return self.load_block(name, col_range=(colId, colId + 1))

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

    def select_block(self, name: str, row_range: (int, int) = None, col_range: (int, int) = None):
        return self.dataset.select_elements(name, row_range=row_range, col_range=col_range)

    def select_row(self, name: str, rowId: int):
        return self.select_block(name, row_range=(rowId, rowId + 1))

    def select_col(self, name: str, colId: int):
        return self.select_block(name, col_range=(colId, colId + 1))

    def matrix_mul(self, A: BufferPiece, B: BufferPiece, C: str):
        self.MUL_counter += A.density * A.shape[0] * B.shape[1]
        return BufferPiece(C, A.range[0], B.range[1], self.dataset.get_density(C))

    # 将计算的结果放入缓存，这不会增加DRAM的访问次数
    def generate_block(self, bufferPiece: BufferPiece):
        self.buffer.load_elements(bufferPiece)
