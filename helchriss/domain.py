
import os
from lark import Lark, Tree, Transformer, v_args
from typing import Set, Tuple, Dict, List, Sequence, Union, Any
from .knowledge import State, Precondition, Effect, Action
from .types import baseType

def carry_func_name(name,expr_nested):
    output = []
    slots = []
    for pos in expr_nested:
        if isinstance(pos, list):
            carried, slot = carry_func_name(name,pos)
            output.append(carried)
            slots.extend(slot)
        elif pos[:2] == "??":
            output.append("{}-{}".format(name, pos[2:]))
            slots.append("{}-{}".format(name, pos[2:]))
        else: output.append(pos)
    return output, slots

class Domain:
    
    def __init__(self, grammar_file = None):
        grammar_file = os.path.join(os.path.dirname(__file__), 'base.grammar') if grammar_file is None else grammar_file
        with open(grammar_file) as file:
            self.lark = Lark(file)
        self.grammar_file = grammar_file
        self.domain_name = None # domain name indication

        self.types = {} # types as a diction, map from the type name to the actual type object

        self.functions = {} # all the function call to the predicates stored, map from the name to actual type

        self.derived = {}

        self.actions = {} # all the actions in the domain

        self.type_constraints = {}

        """
        the actual implementations of the predicates and slots
        during the actual evaluation, if a predicate does not have an implementation, it will use the default method
        as `state.get(predicate_name)`
        """
        self.implementations = {} 
    
    def check_implementation(self) -> bool:
        """
        check if all the slots in the function
        """
        return True
    
    def define_type(self, type_name, parent_name = None):
        if parent_name is None: parent_name = "object"
        self.types[type_name] = parent_name
    
    def define_predicate(self, predicate_name, parameters, output_type):
        self.functions[predicate_name] = {"name":predicate_name,"parameters":parameters, "type":output_type}
    
    def define_action(self, action_name, parameters, precondition, effect):
        """ define symbolic action using the action name, parameters, preconditon and effect, the actual implementation is empty.
        Args:
            action_name: the name of the action
            parameters: the parameters of the action in the form of [?x ?y ... ]
            precondtion: the precondition evaluation function as a binary expression
            effect: the effect expression, notice the predicate could be more than binary
        """
        self.actions[action_name] = Action(action_name, parameters, precondition, effect)

    def define_type_constraint(self,name,controls):
        self.type_constraints[name] = list(controls)
    
    def define_derived(self, name, params, expr):
        carry_name, slots = carry_func_name(name, expr)
        expr_str = to_lambda_expression(carry_name)

        for slot in slots:
            self.implementations[slot] = None
        self.derived[name] = {"name":name, "parameters":params, "expr":expr_str}
    
    def register_slots(self, name, expr_list):
        #TODO: write a version that transforms the lambda expr to nested list
        assert isinstance(expr_list, list),"in the nested list form:{}".format(type(expr_list))
        for pos in expr_list:
            if isinstance(pos, list): [self.register_slots(name, slot) for slot in pos]
            if pos[:2] == "??": self.implementations[name+"-"+pos[2:]] = None

    def print_summary(self):
        print(f"domain:\n  {self.domain_name}")
        print("types:")
        for key in self.types:
            print(f"  {key} - {self.types[key]}")
        print("predicates:")
        for key in self.functions:
            predicate_name = self.functions[key]["name"]
            parameters = self.functions[key]["parameters"]
            output_type = self.functions[key]["type"]
            print(f"  {predicate_name}:{parameters} -> {output_type}");
        print("actions:")
        for key in self.actions:
            atom_action = self.actions[key]
            name = atom_action.action_name
            params = atom_action.parameters
            precond = str(atom_action.precondition)
            effect = str(atom_action.effect)
            print(f" name:{name}\n  params:{params}\n  precond:{precond}\n  effect:{effect}")
        print(self.implementations)

_icc_parser = Domain()

def load_domain_string(domain_string, default_parser = _icc_parser):
    tree = default_parser.lark.parse(domain_string)
    icc_transformer = ICCTransformer(default_parser.grammar_file)
    icc_transformer.transform(tree)
    return icc_transformer.domain

def to_lambda_expression(list):
    if isinstance(list, str): return list
    expr = f"({list[0]}"
    for cont in list[1:]:
        expr += " {}".format(to_lambda_expression(cont))
    expr += ")"
    return expr

class ICCTransformer(Transformer):
    def __init__(self, grammar_file = None):
        super().__init__()
        self.domain = Domain(grammar_file)

    def domain_definition(self, domain_name):
        domain_name = domain_name[0]
        self.domain.domain_name = str(domain_name)
    
    """
    predicate definition handle part
    """
    def predicate_definition(self, args):
        predicate_name = args[0]
        if len(args) == 2:
            parameters = []
        else:
            parameters = args[1]

        output_type = args[-1]
        self.domain.define_predicate(predicate_name, parameters, output_type)
    
    def predicate_name(self, args):return str(args[0])

    def parameters(self, args):return [str(arg) for arg in args]
    
    def parameter(self, args):return str(args[0])
    
    def object_type_name(self, args):return str(args[0])
    
    """
    type definition handler part, use the args in the format of ?x ?y...
    """
    def type_definition(self, args):
        type_name = args[0]
        if len(args) == 1:self.domain.define_type(type_name)
        if len(args) == 2:self.domain.define_type(type_name, args[1])
    
    def value_type_name(self, args):
        return args[0]
    
    def vector_type_name(self, args):
        print(args)
        return args
    
    def type_name(self, args):
        return str(args[0])
  
    def parent_type_name(self, args):
        return args[0]
    
    def typed_variable(self, args):
        return str(args[0]) + "-" + args[1]
    
    def vector_type_name(self, args):
        vector_choice = args[0]
        vector_size = [str(num).replace("'","") for num in args[1:]]
        return f"vector[{vector_choice},{vector_size}]"
    
    def list_type_name(self, args):
        return f"List[{args[0]}]"
    
    def vector_choice(self, args):return args[0]
    
    def vector_size(self, args):return args[0]
    
    def number(self,args):return str(args[0])

    """
    action definition handler part, use the format of
    (
        action: name
        parameters: ?x ?y ...
        precondition: (and)
        effect: (and)
    )
    """
    def action_definition(self, args):
        #print(args[0])
        #print(args[1])
        #print(args[2])
        #print(args[3])

        precond = to_lambda_expression(args[2])
        effect = to_lambda_expression(args[3])
        self.domain.define_action(args[0], args[1], precond, effect)
    
    def action_name(self, args): return str(args[0])

    def precondition(self, args): return args[0]

    def effect(self, args): return args[0]

    """
    function and expression calls
    """
    def function_call(self, args):
        return args
    
    def function_name(self,args):
        return args[0]
    
    def variable(self, args):
        return str(args[0])
    
    def VARNAME(self, args):
        return str(args)

    def RULE(self, args):
        return str(args[0])

    def CONSTNAME(self, args):
        return str(args)
    
    def constant(self, args):
        return [str(arg) for arg in args]

    """Handle the part where functions are slots"""
    def slot(self,args):
        return str(args[0] + args[1][0])
    
    def SLOT(self, args):return str(args)
    
    def slot_name(self,args):
        return args

    """Functional Handler"""    
    def functional_name(self, args):
        return  str(args[0])
    
    def functional_definition(self, args):
        functional_name = args[0]

    """Type Constraints Definitions"""
    def constraint_definition(self, args):
        type_name = args[0]
        arg_controls = args[1:]
        self.domain.define_type_constraint(type_name, arg_controls)

    """Derived predicates Definitions"""
    def derived_definition(self, args):
        self.domain.define_derived(args[0], args[1],args[2])
    
    def derived_name(self,args):
        return str(args[0])
    
