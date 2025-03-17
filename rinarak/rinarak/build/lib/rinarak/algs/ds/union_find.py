class UnionFind:
    def __init__(self, elements):
        self.father = {}
        for e in elements:self.father[e] = e
        self.sizes = []
        self.depths = []

    def union(self, x, y):
        father_x = self.find(x)
        father_y = self.find(y)
        if father_x == father_y:
            return 
        self.father[father_x] = father_y
    
    def find(self, x):
        if self.father[x] == x: return x 
        fa = self.find(self.father[x])
        self.father[x] = fa
        return self.father[x]

if __name__ == "__main__":
    xs = ["A","B","C","D","E"]
    ufs = UnionFind(xs)

    ufs.union("A","B")
    print(ufs.find("A"))
    print(ufs.find("B"))
    print(ufs.find("C"))
    print(ufs.find("D"))
    ufs.union("A","D")
    print(ufs.find("D"))
    print(ufs.find("A"))
    print(ufs.find("B"))
    print(ufs.find("C"))