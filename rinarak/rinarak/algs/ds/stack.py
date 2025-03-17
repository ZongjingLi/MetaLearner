class Stack:
    def __init__(self):
        self.values = []
    
    def add(self, x):
        self.values.append(x)
        return self.values[-1]

    def pop(self, x):
        self.values[:len(self.values) - 1]

    def top(self):return self.values[-1]