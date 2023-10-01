
special_tokens = {"In":"","Not":"!","And":"^"}

class PredicateExpression:
    special_tokens = special_tokens
    def __init__(self, special_tokens):
        self.readable = True
        self.precondition = None
        self.effect = None
        self.action = None

    def check_applicable(self):return