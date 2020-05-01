import sys
import unittest 
import gym
import torch
import numpy as np
import random
import cv2
import time
sys.path.append("./")
from utils.pmdb import SegTree

class TestCase(unittest.TestCase):
 
    def test_naive_update_full_prior(self, length=1024**2):
        prior_lst = [0 for _ in range(2*length-1)]
        prior_lst[-length:] = [random.random() for _ in range(length)]
        t0 = time.time()
        print(len(prior_lst))
        for i in range(2*length-2, 1, -2):
            prior_lst[(i-1)//2] = prior_lst[i] + prior_lst[i-1]
        t1 = time.time()
        assert abs(prior_lst[0] - sum(prior_lst[-length:])) < 1e-5
        print("\nnaive_update_full_prior: {}s".format(t1-t0))
        return t1-t0

    def test_naive_update_batch_prior(self, length=1024**2, batch=2560):
        prior_lst = [0 for _ in range(2*length-1)]
        prior_lst[-length:] = [random.random() for _ in range(length)]
        for i in range(2*length-2, 1, -2):
            prior_lst[(i-1)//2] = prior_lst[i] + prior_lst[i-1]
        
        idxs = [random.randint(length-1, 2*length-2) for _ in range(batch)]
        total = prior_lst[0]
        for idx in idxs: 
            x = random.random()
            total += x-prior_lst[idx]
            prior_lst[idx] = x
        
        t0 = time.time()
        for idx in idxs:
            top = idx
            while top > 0:
                top = (top-1)//2
                left = 2*top+1
                prior_lst[top] = prior_lst[left] + prior_lst[left+1]
        t1 = time.time()
        print("\ntest_naieve_update_batch_prior({}) {}s".format(batch, t1-t0))
        assert abs(prior_lst[0] - total) < 1e-5
        return t1-t0

    def test_naive_query_batch_prior(self, length=1024**2, batch=256):
        prior_lst = [0 for _ in range(2*length-1)]
        prior_lst[-length:] = [random.random() for _ in range(length)]
        for i in range(2*length-2, 1, -2):
            prior_lst[(i-1)//2] = prior_lst[i] + prior_lst[i-1]  
        
        qs = [random.random()*(prior_lst[0]-1e-5) for _ in range(batch)]
        idxs = []
        t0 = time.time()
        l = len(prior_lst)
        for q in qs:
            top = 0
            left = 2*top+1
            while left < l:
                left_value = prior_lst[left]
                if q < left_value:
                    top = left
                else:
                    q -= left_value
                    top = left+1
                left = 2*top+1
            idxs.append(top-length+1)
        t1 = time.time()
        print("\ntest_naive_query_batch_prior({}) {}s".format(batch, t1-t0))
        for q, idx in zip(qs, idxs):
            assert sum(prior_lst[-length:-length+idx]) < q and sum(prior_lst[-length:-length+idx+1]) > q, \
                "{} {} {}".format(sum(prior_lst[-length:-length+idx]), sum(prior_lst[-length:-length+idx+1]), q)
        return t1-t0

    def init_segtree(self, length=1024**2, num=1024**2):
        tree = SegTree(length)
        rand_float = [random.random() for _ in range(num)]
        t0 = time.time()
        tree.append(rand_float)
        t1 = time.time()
        assert abs(tree.total() - sum(rand_float[-min(length, num):])) < 1e-5
        print("\ninit_segtree: {}s".format(t1-t0))
        return tree   

    def insert_segtree(self, tree, num=2560):
        rand_float = [random.random() for _ in range(num)]
        sum0 = tree.total()
        sum1 = sum0
        for i, x in enumerate(rand_float):
            sum1 -= tree[(i+tree.index)%tree.max_num]
            sum1 += x
        t0 = time.time()
        tree.append(rand_float)
        t1 = time.time()
        print("\ninsert({}) {}s".format(num, t1-t0))
        assert abs(tree.total()-sum1) < 1e-5

    def update_segtree(self, tree, num=2560):
        rand_float = [random.random() for _ in range(num)]
        idxs = random.sample(list(range(len(tree))), num)
        sum0 = tree.total()
        sum1 = sum0
        for i, x in zip(idxs, rand_float):
            sum1 -= tree[i]
            sum1 += x    
        t0 = time.time()
        tree.update(idxs, rand_float)
        t1 = time.time()  
        print("\nupdate({}) {}s".format(num, t1-t0))     
        assert abs(tree.total()-sum1) < 1e-5       

    def query_segtree(self, tree, num=2560):
        total = tree.total()
        values = [random.random()*total for _ in range(num)]
        t0 = time.time()
        idxs = tree.find(values)
        t1 = time.time()
        print("\nfind({}) {}s".format(num, t1-t0)) 
        for value, idx in zip(values, idxs):
            sum0 = 0
            for i in range(idx):
                sum0 += tree[i]
            sum1 = sum0 + tree[idx]
            assert sum0 < value and sum1 > value
        

    def init_0(self):
        return self.init_segtree(length=1000**2, num=1024)

    def init_1(self):
        return self.init_segtree(length=1000**2, num=1024**2)

    def insert_0(self):
        tree = self.init_segtree(length=1000**2, num=1024)
        self.insert_segtree(tree, num=2560)

    def insert_1(self):
        tree = self.init_segtree(length=1000**2, num=1024**2)
        self.insert_segtree(tree, num=2560) 

    def update_0(self):     
        tree = self.init_segtree(length=1000**2, num=2560)
        self.update_segtree(tree, num=2560) 

    def update_1(self):     
        tree = self.init_segtree(length=1000**2, num=1024**2)
        self.update_segtree(tree, num=2560) 

    def find_0(self):
        tree = self.init_segtree(length=1000**2, num=1024**2)
        self.query_segtree(tree, num=256)



def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestCase("test_naive_update_full_prior"))
    # suite.addTest(TestCase("test_naive_update_batch_prior"))
    # suite.addTest(TestCase("test_naive_query_batch_prior"))
    suite.addTest(TestCase("init_0"))
    suite.addTest(TestCase("init_1"))
    suite.addTest(TestCase("insert_0"))
    suite.addTest(TestCase("insert_1"))
    suite.addTest(TestCase("update_0"))
    suite.addTest(TestCase("update_1"))
    suite.addTest(TestCase("find_0"))
    
    return suite

if __name__ == "__main__":
    unittest.main(defaultTest = 'suite')  #在主函数中调用全局方法