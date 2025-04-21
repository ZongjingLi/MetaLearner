'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-11-26 15:40:14
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-11-26 15:41:17
 # @ Description: This file is distributed under the MIT license.
 '''

from dataclasses import dataclass
from webbrowser import get

from typing import TypeVar, List, Callable, Iterator, Tuple, Set
import heapq as hq

__all__ = ['SearchNode', 'QueueNode', 'run_heuristic_search', 'backtrace_plan']

State = TypeVar("State")
Action = TypeVar("Action")

@dataclass
class SearchNode(object):
    """SearchNode represents a node that contains :state, parent, action, cost, g"""
    
    state: State
    """The current state"""

    parentNode: 'SearchNode'
    """The parent node of the current node"""

    action: Action
    """The action taken that leads to this node"""

    cost: float
    """The cost from root node to current node"""

    g: float
    """The estimated cost to go"""

@dataclass
class QueueNode(object):
    """A node in the Queue that contains: priority, node """

    priority: float
    """Priority of the Node"""

    node: 'SearchNode'
    """The search Node"""

    @property
    def state(self):return self.node.state

    def __iter__(self):
        yield self.priority
        yield self.node
    
    def __lt__(self, other):
        return self.priority < other.priority
    
def run_heuristic_search(
    init_state: State,
    check_goal: Callable[[State], bool],
    get_priority: Callable[[State, int], float],
    get_successors: Callable[[State], Iterator[Tuple[Action, State, float]]],
    check_visited: bool = True,
    max_expansions: int = 100000,
    max_depth: int = 100000
):
    """
    A generic implementation for the heuristic search
    Args:
        init_state: the inital state to search
        check_goal: a function mapping from state to bool, returning True if the goal is satisfied at the state
        get_priority: a function that maps from state to float that represents the priority of the state
        get_successors: a function map the state to an iterator of (action, state, cost) tuple
        check_visited: whether to check if the state has been visisted, False means the state is not hashable
        max_expansions: the maximum number of expansions.
        max_depth: the maximum depth of the search tree.
    Returns:
        a tuple of (state sequence, action_sequence, cost_sequence, num_expansions)
    Raises:
        ValueError: if the search fails
    """
    queue: List[QueueNode] = list()
    visited: Set[State] = set()
    def push_node(node: SearchNode):
        hq.heappush(queue, QueueNode(get_priority(node.state, node.g), node))
        if check_visited:
            visited.add(node.state)
    root_node = SearchNode(state = init_state, parentNode = None, action = None, cost = None, g = 0)
    push_node(root_node)
    num_expansions = 0
    #print(root_node.state.state["red"], root_node.state.state["green"])
    
    while len(queue) > 0 and num_expansions < max_expansions:
        prioirty, node = hq.heappop(queue)
        #print(node.state.state["red"], node.state.state["green"])
        if node.g > max_depth:
            raise RuntimeError()
        
        if check_goal(node.state):
            return backtrace_plan(node, num_expansions)
        num_expansions += 1
        for action, child_state, cost in get_successors(node.state):
            if check_visited and child_state in visited:
                continue
            
            child_node = SearchNode(state = child_state, parentNode=node, action=action, cost=cost, g = node.g + cost)
            #print(node.state.state["red"], node.state.state["green"])
            push_node(child_node)
    
    raise RuntimeError("Failed to find a plan (maximum expansion reached)")

def backtrace_plan(node: SearchNode, nr_expansions: int) -> Tuple[List[State], List[Action], List[float], int]:
    """Backtrace the plan from the goal node.

    Args:
        node: a search node where the goal is satisfied.
        nr_expansions: the number of expansions. This value will be returned by this function.

    Returns:
        a tuple of (state_sequence, action_sequence, cost_sequence, nr_expansions).
    """
    state_sequence = []
    action_sequence = []
    cost_sequence = []
    
    while node.parentNode is not None:
        state_sequence.insert(0, node.state)
        action_sequence.insert(0, node.action)
        cost_sequence.insert(0, node.cost)
        node = node.parentNode
    state_sequence.insert(0, node.state)
    return state_sequence, action_sequence, cost_sequence, nr_expansions