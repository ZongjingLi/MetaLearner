import numpy as np

def swap(a, b):return b,a

def real_comp(a, b, lesser = True):
    if lesser:return a<b
    else:return b<a

class Treap:
    def __init__(self, data, comparator = None, lesser=True):
        self.data = []
        # [Compare Node]
        self.comparator = None
        self.lesser = lesser

        for node in data: self.add(node)

    
    def add(self, node):
        self.data.append(node)
        if self.comparator is not None:
            compare = self.comparator
        else: compare = real_comp
        curr_idx = len(self.data) - 1

        curr_node = self.data[curr_idx]
        parent_node = self.data[curr_idx // 2] 
        
        while (compare(curr_node, parent_node,lesser = self.lesser)):
            tmp = self.data[curr_idx]
            self.data[curr_idx] = self.data[curr_idx//2] 
            self.data[curr_idx//2]  = tmp

            curr_idx //= 2
            curr_node = self.data[curr_idx]
            parent_node = self.data[curr_idx // 2]      

        return -1

    def top(self, k = None):
        if k is not None:return self.data[:k]
        else:return self.data[0]
    
    def pop(self):
        return self.data.__delitem__(0)

if __name__ == "__main__":
    data = [100,3,44,33,22,33,33,33,33]
    treap = Treap(data, lesser = 0)

    print(treap.top(6))

    treap.pop()

    print(treap.top(6))

    treap.pop()

    print(treap.top(6))
