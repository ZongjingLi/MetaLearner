class Queue:
    def __init__(self, max_size):
        self.values = [None] * max_size
        self.front = 0
        self.rear = 1
    
    def add(self, x):
        self.values[self.rear] = x
        self.rear += 1
    
    def pop(self):self.front += 1

    def top(self):return self.values[self.front+1]

    def is_empty(self):return self.front+1 == self.rear

    def empty(self):
        self.front = 0
        self.rear  = 1

if __name__ == "__main__":
    q = Queue(100)
    q.add(4)
    q.add(3)
    q.add(5)
    for v in q.values:
        if v is not None:print(v)
    print(q.top())
    q.pop()
    print(q.top())