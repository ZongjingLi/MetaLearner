
special_tokens = {"In":"","Not":"!","And":"^"}

class PredicateExpression:
    special_tokens = special_tokens
    def __init__(self, special_tokens):
        self.readable = True

    def parse(predicate):
        """
        input: predicate representation that is not so human readable
        output: normal format that is human readable
        """
        return 
