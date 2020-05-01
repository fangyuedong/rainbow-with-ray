import os, sys
import math
sys.path.append("./")
from utils.mmdb import Mmdb

class SegTree():
    def __init__(self, max_num=1000**2):
        self.max_num = max_num
        self.index = 0
        self.history_len = 0
        self.struct_len = 2**math.ceil(math.log(max_num, 2))
        self.sum_tree = [0.0 for _ in range(2*self.struct_len-1)]

    def _propagate(self, idx, value):
        parent = idx
        self.sum_tree[parent] = value
        while parent > 0:
            parent = (parent-1)//2
            left = 2*parent+1
            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[left+1]
        
    def update(self, idxs, values):
        for idx, value in zip(idxs, values):self._propagate(idx+self.struct_len-1, value)

    def append(self, values):
        for value in values:
            self._propagate(self.index+self.struct_len-1, value)
            self.index = (self.index+1)%self.max_num
        self.history_len += len(values)

    def _retrieve(self, value):
        assert value < self.sum_tree[0]
        top = 0
        left = 2*top+1
        l = len(self.sum_tree)
        while left < l:
            left_value = self.sum_tree[left]
            if value < left_value:
                top = left
            else:
                value -= left_value
                top = left+1
        return top

    def find(self, values):
        return [self._retrieve(value)-self.struct_len+1 for value in values]

    def __getitem__(self, idx):
        return self.sum_tree[idx+self.struct_len-1]
    
    def __len__(self):
        return min(self.history_len, self.max_num)

    def total(self):
        return self.sum_tree[0]




    

        