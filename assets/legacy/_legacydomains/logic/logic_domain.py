'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-03-03 16:12:06
 # @Modified by: Your name
 # @Modified time: 2025-03-03 16:12:07
'''

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any

from domains.utils import (
    load_domain_string,
    domain_parser,
    build_domain_executor
)
from rinarak.knowledge.executor import CentralExecutor
from rinarak.program import Primitive, arrow
from rinarak.dsl.logic_types import boolean
from rinarak.types import treal, tvector

logic_domain_str = """
(domain Logic)
(:type
    boolean - vector[float, 1] ;; abstract repr of function
)
(:predicate
    and ?x-boolean ?y-boolean -> boolean
    or ?x-boolean ?y-boolean -> boolean
    not ?x-boolean -> boolean
    implies ?x-boolean ?y-boolean -> boolean
    equivalent ?x-boolean ?y-boolean -> boolean
    xor ?x-boolean ?y-boolean -> boolean
)
"""

ltl_domain_str = """
(domain LTL)
(:type
    boolean - vector[float, 1]
    time - float
    state - vector[float, 1]
)
(:predicate
    eventually ?x-boolean -> boolean
    always ?x-boolean -> boolean
    next ?x-boolean -> boolean
    until ?x-boolean ?y-boolean -> boolean
    release ?x-boolean ?y-boolean -> boolean
    since ?x-boolean ?y-boolean -> boolean
    previously ?x-boolean -> boolean
    at ?x-boolean ?t-time -> boolean
    holds_at ?x-state ?t-time -> boolean
)
"""

epistemic_domain_str = """
(domain Epistemic)
(:type
    boolean - vector[float, 1]
    agent - integer
    knowledge - vector[float, 1]
)
(:predicate
    knows ?a-agent ?p-boolean -> boolean
    possible ?a-agent ?p-boolean -> boolean
    common_knowledge ?p-boolean -> boolean
    believes ?a-agent ?p-boolean -> boolean
    confident ?a-agent ?p-boolean -> boolean
    uncertain ?a-agent ?p-boolean -> boolean
    public_announcement ?p-boolean -> boolean
    private_announcement ?a-agent ?p-boolean -> boolean
)
"""

class LogicDomain:
    """Handler for logic predicates and boolean operations.
    
    Implements differentiable predicates for boolean logic operations
    with smooth transitions controlled by temperature.
    """
    
    def __init__(self, temperature: float = 0.01, epsilon: float = 1e-6):
        """Initialize logic domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations, controls transition sharpness
            epsilon: Small value for numerical stability
        """
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid with temperature scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            Sigmoid-scaled tensor
        """
        return torch.sigmoid(x / self.temperature)
    
    def and_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable logical AND operation.
        
        Args:
            x: [B1, 1] tensor of first boolean values
            y: [B2, 1] tensor of second boolean values
            
        Returns:
            [B1, B2] tensor of AND operation results
        """
        x_exp = x.unsqueeze(1)  # [B1, 1, 1]
        y_exp = y.unsqueeze(0)  # [1, B2, 1]
        return torch.min(x_exp, y_exp).squeeze(-1)
    
    def or_op(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable logical OR operation.
        
        Args:
            x: [B1, 1] tensor of first boolean values
            y: [B2, 1] tensor of second boolean values
            
        Returns:
            [B1, B2] tensor of OR operation results
        """
        x_exp = x.unsqueeze(1)  # [B1, 1, 1]
        y_exp = y.unsqueeze(0)  # [1, B2, 1]
        return torch.max(x_exp, y_exp).squeeze(-1)
    
    def not_op(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable logical NOT operation.
        
        Args:
            x: [B, 1] tensor of boolean values
            
        Returns:
            [B] tensor of NOT operation results
        """
        return 1.0 - x
    
    def implies(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable logical implication (x → y).
        
        Args:
            x: [B1, 1] tensor of first boolean values
            y: [B2, 1] tensor of second boolean values
            
        Returns:
            [B1, B2] tensor of implication results
        """
        not_x = self.not_op(x)
        not_x_exp = not_x.unsqueeze(1)  # [B1, 1, 1]
        y_exp = y.unsqueeze(0)  # [1, B2, 1]
        return torch.max(not_x_exp, y_exp).squeeze(-1)
    
    def equivalent(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable logical equivalence (x ↔ y).
        
        Args:
            x: [B1, 1] tensor of first boolean values
            y: [B2, 1] tensor of second boolean values
            
        Returns:
            [B1, B2] tensor of equivalence results
        """
        x_exp = x.unsqueeze(1)  # [B1, 1, 1]
        y_exp = y.unsqueeze(0)  # [1, B2, 1]
        return 1.0 - torch.abs(x_exp - y_exp).squeeze(-1)
    
    def xor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable logical XOR operation.
        
        Args:
            x: [B1, 1] tensor of first boolean values
            y: [B2, 1] tensor of second boolean values
            
        Returns:
            [B1, B2] tensor of XOR operation results
        """
        x_exp = x.unsqueeze(1)  # [B1, 1, 1]
        y_exp = y.unsqueeze(0)  # [1, B2, 1]
        return torch.abs(x_exp - y_exp).squeeze(-1)
        
    def setup_predicates(self, executor: CentralExecutor):
        """Setup all logic predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        boolean_type = tvector(treal, 1)  # Boolean representation
        
        executor.update_registry({
            "and": Primitive(
                "and",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.and_op(x["state"], y["state"])}
            ),
            
            "or": Primitive(
                "or",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.or_op(x["state"], y["state"])}
            ),
            
            "not": Primitive(
                "not",
                arrow(boolean_type, boolean_type),
                lambda x: {**x, "end": self.not_op(x["state"])}
            ),
            
            "implies": Primitive(
                "implies",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.implies(x["state"], y["state"])}
            ),
            
            "equivalent": Primitive(
                "equivalent",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.equivalent(x["state"], y["state"])}
            ),
            
            "xor": Primitive(
                "xor",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.xor(x["state"], y["state"])}
            )
        })

class LTLDomain:
    """Handler for Linear Temporal Logic predicates.
    
    Implements differentiable predicates for temporal reasoning
    with smooth transitions controlled by temperature.
    """
    
    def __init__(self, temperature: float = 0.01, time_horizon: float = 10.0, 
                 epsilon: float = 1e-6):
        """Initialize LTL domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations, controls transition sharpness
            time_horizon: Maximum time window for temporal operators
            epsilon: Small value for numerical stability
        """
        self.temperature = temperature
        self.time_horizon = time_horizon
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid with temperature scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            Sigmoid-scaled tensor
        """
        return torch.sigmoid(x / self.temperature)
    
    def eventually(self, x: torch.Tensor, time_points: torch.Tensor = None) -> torch.Tensor:
        """Differentiable 'eventually' temporal operator.
        
        Args:
            x: [B, T] tensor of boolean values over time
            time_points: Optional tensor of time points
            
        Returns:
            [B] tensor of eventually operation results
        """
        # By default, maximum over time dimension
        return torch.max(x, dim=1)[0]
    
    def always(self, x: torch.Tensor, time_points: torch.Tensor = None) -> torch.Tensor:
        """Differentiable 'always' temporal operator.
        
        Args:
            x: [B, T] tensor of boolean values over time
            time_points: Optional tensor of time points
            
        Returns:
            [B] tensor of always operation results
        """
        # By default, minimum over time dimension
        return torch.min(x, dim=1)[0]
    
    def next(self, x: torch.Tensor, time_points: torch.Tensor = None) -> torch.Tensor:
        """Differentiable 'next' temporal operator.
        
        Args:
            x: [B, T] tensor of boolean values over time
            time_points: Optional tensor of time points
            
        Returns:
            [B, T-1] tensor of next operation results
        """
        # Shift one step forward in time
        return x[:, 1:]
    
    def until(self, x: torch.Tensor, y: torch.Tensor, 
             time_points: torch.Tensor = None) -> torch.Tensor:
        """Differentiable 'until' temporal operator.
        
        Args:
            x: [B, T] tensor of first boolean values over time
            y: [B, T] tensor of second boolean values over time
            time_points: Optional tensor of time points
            
        Returns:
            [B] tensor of until operation results
        """
        # Implementation of x U y
        B, T = x.shape
        result = torch.zeros(B, device=self.device)
        
        for i in range(T):
            # y holds at time i
            y_i = y[:, i]
            
            # x holds continuously until time i
            x_until_i = torch.ones(B, device=self.device)
            for j in range(i):
                x_until_i = torch.min(x_until_i, x[:, j])
            
            # result |= (y_i && x_until_i)
            current = torch.min(y_i, x_until_i)
            result = torch.max(result, current)
            
        return result
    
    def release(self, x: torch.Tensor, y: torch.Tensor, 
               time_points: torch.Tensor = None) -> torch.Tensor:
        """Differentiable 'release' temporal operator.
        
        Args:
            x: [B, T] tensor of first boolean values over time
            y: [B, T] tensor of second boolean values over time
            time_points: Optional tensor of time points
            
        Returns:
            [B] tensor of release operation results
        """
        # Implementation of x R y
        B, T = x.shape
        result = torch.ones(B, device=self.device)
        
        for i in range(T):
            # y holds at time i OR x holds before or at time i
            y_i = y[:, i]
            x_before_i = torch.zeros(B, device=self.device)
            
            for j in range(i+1):
                x_before_i = torch.max(x_before_i, x[:, j])
                
            current = torch.max(y_i, x_before_i)
            result = torch.min(result, current)
            
        return result
    
    def since(self, x: torch.Tensor, y: torch.Tensor, 
             time_points: torch.Tensor = None) -> torch.Tensor:
        """Differentiable 'since' temporal operator.
        
        Args:
            x: [B, T] tensor of first boolean values over time
            y: [B, T] tensor of second boolean values over time
            time_points: Optional tensor of time points
            
        Returns:
            [B, T] tensor of since operation results
        """
        # Implementation of x S y
        B, T = x.shape
        result = torch.zeros((B, T), device=self.device)
        
        for t in range(T):
            for i in range(t+1):
                # y holds at time i
                y_i = y[:, i]
                
                # x holds continuously from i+1 to t
                x_since_i = torch.ones(B, device=self.device)
                for j in range(i+1, t+1):
                    x_since_i = torch.min(x_since_i, x[:, j])
                
                # result |= (y_i && x_since_i)
                current = torch.min(y_i, x_since_i)
                result[:, t] = torch.max(result[:, t], current)
                
        return result
    
    def previously(self, x: torch.Tensor, time_points: torch.Tensor = None) -> torch.Tensor:
        """Differentiable 'previously' temporal operator.
        
        Args:
            x: [B, T] tensor of boolean values over time
            time_points: Optional tensor of time points
            
        Returns:
            [B, T] tensor of previously operation results
        """
        # Shift one step backward in time
        result = torch.zeros_like(x)
        result[:, 1:] = x[:, :-1]
        return result
    
    def at(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Value of boolean expression at specific time.
        
        Args:
            x: [B, T] tensor of boolean values over time
            t: Time index
            
        Returns:
            [B] tensor of values at specified time
        """
        t_idx = t.long()
        return x[:, t_idx]
    
    def holds_at(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Check if state holds at specific time.
        
        Args:
            state: [B, D] tensor of state values
            t: Time index
            
        Returns:
            [B] tensor of state validity at specified time
        """
        # Simple implementation - returns truth value of state at time t
        t_idx = t.long()
        return state[:, t_idx]
        
    def setup_predicates(self, executor: CentralExecutor):
        """Setup all LTL predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        boolean_type = tvector(treal, 1)  # Boolean representation
        time_type = treal  # Time point
        state_type = tvector(treal, 1)  # State representation
        
        executor.update_registry({
            "eventually": Primitive(
                "eventually",
                arrow(boolean_type, boolean_type),
                lambda x: {**x, "end": self.eventually(x["state"])}
            ),
            
            "always": Primitive(
                "always",
                arrow(boolean_type, boolean_type),
                lambda x: {**x, "end": self.always(x["state"])}
            ),
            
            "next": Primitive(
                "next",
                arrow(boolean_type, boolean_type),
                lambda x: {**x, "end": self.next(x["state"])}
            ),
            
            "until": Primitive(
                "until",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.until(x["state"], y["state"])}
            ),
            
            "release": Primitive(
                "release",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.release(x["state"], y["state"])}
            ),
            
            "since": Primitive(
                "since",
                arrow(boolean_type, arrow(boolean_type, boolean_type)),
                lambda x: lambda y: {**x, "end": self.since(x["state"], y["state"])}
            ),
            
            "previously": Primitive(
                "previously",
                arrow(boolean_type, boolean_type),
                lambda x: {**x, "end": self.previously(x["state"])}
            ),
            
            "at": Primitive(
                "at",
                arrow(boolean_type, arrow(time_type, boolean_type)),
                lambda x: lambda t: {**x, "end": self.at(x["state"], t["state"])}
            ),
            
            "holds_at": Primitive(
                "holds_at",
                arrow(state_type, arrow(time_type, boolean_type)),
                lambda s: lambda t: {**s, "end": self.holds_at(s["state"], t["state"])}
            )
        })

class EpistemicDomain:
    def __init__(self, temperature: float = 0.01, num_agents: int = 5, 
                 epsilon: float = 1e-6):
        """Initialize epistemic domain with parameters.
        
        Args:
            temperature: Temperature for smooth operations, controls transition sharpness
            num_agents: Number of agents in the epistemic model
            epsilon: Small value for numerical stability
        """
        self.temperature = temperature
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.knowledge_base = torch.zeros((num_agents, 100), device=self.device)
        
    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid with temperature scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            Sigmoid-scaled tensor
        """
        return torch.sigmoid(x / self.temperature)
    
    def knows(self, agent: torch.Tensor, proposition: torch.Tensor) -> torch.Tensor:
        """Check if agent knows proposition.
        
        Args:
            agent: [B1] tensor of agent IDs
            proposition: [B2, 1] tensor of proposition values
            
        Returns:
            [B1, B2] tensor of knowledge scores
        """
        agent_idx = agent.long()
        agent_knowledge = self.knowledge_base[agent_idx]
        
        # Expand dimensions for broadcasting
        agent_knowledge = agent_knowledge.unsqueeze(1)  # [B1, 1, K]
        prop_embedding = proposition.unsqueeze(0)  # [1, B2, 1]
        
        # Compute similarity between knowledge and proposition
        similarity = 1.0 - torch.abs(agent_knowledge - prop_embedding.unsqueeze(-1)).mean(-1)
        
        # Knowledge threshold
        return self._sigmoid((similarity - 0.8) / self.temperature)
    
    def possible(self, agent: torch.Tensor, proposition: torch.Tensor) -> torch.Tensor:
        """Check if proposition is possible according to agent's knowledge.
        
        Args:
            agent: [B1] tensor of agent IDs
            proposition: [B2, 1] tensor of proposition values
            
        Returns:
            [B1, B2] tensor of possibility scores
        """
        # If agent doesn't know ~p, then p is possible
        not_p = 1.0 - proposition
        knows_not_p = self.knows(agent, not_p)
        return 1.0 - knows_not_p
    
    def common_knowledge(self, proposition: torch.Tensor) -> torch.Tensor:
        """Check if proposition is common knowledge among all agents.
        
        Args:
            proposition: [B, 1] tensor of proposition values
            
        Returns:
            [B] tensor of common knowledge scores
        """
        all_knows = torch.ones((proposition.shape[0]), device=self.device)
        
        for a in range(self.num_agents):
            agent_tensor = torch.tensor([a], device=self.device).expand(proposition.shape[0])
            knows_a = self.knows(agent_tensor, proposition)
            all_knows = torch.min(all_knows, knows_a)
            
        return all_knows
    
    def believes(self, agent: torch.Tensor, proposition: torch.Tensor) -> torch.Tensor:
        """Check if agent believes proposition (weaker than knowledge).
        
        Args:
            agent: [B1] tensor of agent IDs
            proposition: [B2, 1] tensor of proposition values
            
        Returns:
            [B1, B2] tensor of belief scores
        """
        agent_idx = agent.long()
        agent_knowledge = self.knowledge_base[agent_idx]
        
        # Expand dimensions for broadcasting
        agent_knowledge = agent_knowledge.unsqueeze(1)  # [B1, 1, K]
        prop_embedding = proposition.unsqueeze(0)  # [1, B2, 1]
        
        # Compute similarity between knowledge and proposition
        similarity = 1.0 - torch.abs(agent_knowledge - prop_embedding.unsqueeze(-1)).mean(-1)
        
        # Belief threshold (lower than knowledge)
        return self._sigmoid((similarity - 0.5) / self.temperature)
    
    def confident(self, agent: torch.Tensor, proposition: torch.Tensor) -> torch.Tensor:
        """Check if agent is confident about proposition.
        
        Args:
            agent: [B1] tensor of agent IDs
            proposition: [B2, 1] tensor of proposition values
            
        Returns:
            [B1, B2] tensor of confidence scores
        """
        belief = self.believes(agent, proposition)
        # Confidence is stronger belief
        return self._sigmoid((belief - 0.7) / self.temperature)
    
    def uncertain(self, agent: torch.Tensor, proposition: torch.Tensor) -> torch.Tensor:
        """Check if agent is uncertain about proposition.
        
        Args:
            agent: [B1] tensor of agent IDs
            proposition: [B2, 1] tensor of proposition values
            
        Returns:
            [B1, B2] tensor of uncertainty scores
        """
        belief_p = self.believes(agent, proposition)
        belief_not_p = self.believes(agent, 1.0 - proposition)
        
        # Uncertainty is when belief in p and ~p are both moderate
        middle_belief = self._sigmoid((0.5 - torch.abs(belief_p - 0.5)) / self.temperature)
        middle_belief_not = self._sigmoid((0.5 - torch.abs(belief_not_p - 0.5)) / self.temperature)
        
        return torch.min(middle_belief, middle_belief_not)
    
    def public_announcement(self, proposition: torch.Tensor) -> torch.Tensor:
        """Make public announcement of proposition (updates all agents' knowledge).
        
        Args:
            proposition: [B, 1] tensor of proposition values
            
        Returns:
            [B] tensor indicating success
        """
        # Update knowledge base for all agents
        prop_embedding = proposition.unsqueeze(1)  # [B, 1, 1]
        
        # Simulate knowledge update (in practice, would modify self.knowledge_base)
        # Here we just return success indicator
        return torch.ones((proposition.shape[0]), device=self.device)
    
    def private_announcement(self, agent: torch.Tensor, 
                            proposition: torch.Tensor) -> torch.Tensor:
        """Make private announcement to specific agent.
        
        Args:
            agent: [B1] tensor of agent IDs
            proposition: [B2, 1] tensor of proposition values
            
        Returns:
            [B1, B2] tensor indicating success
        """
        # Update knowledge base for specific agent
        agent_idx = agent.long()
        prop_embedding = proposition.unsqueeze(1)  # [B, 1, 1]
        
        # Simulate knowledge update (in practice, would modify self.knowledge_base)
        # Here we just return success indicator
        return torch.ones((agent.shape[0], proposition.shape[0]), device=self.device)
        
    def setup_predicates(self, executor: CentralExecutor):
        """Setup all epistemic predicates with type signatures.
        
        Args:
            executor: Executor instance to register predicates with
        """
        # Define base types
        boolean_type = tvector(treal, 1)  # Boolean representation
        agent_type = tvector(treal, 1)
        knowledge_type = tvector(treal, 100)  # Knowledge representation
        
        executor.update_registry({
            "knows": Primitive(
                "knows",
                arrow(agent_type, arrow(boolean_type, boolean_type)),
                lambda a: lambda p: {**a, "end": self.knows(a["state"], p["state"])}
            ),
            
            "possible": Primitive(
                "possible",
                arrow(agent_type, arrow(boolean_type, boolean_type)),
                lambda a: lambda p: {**a, "end": self.possible(a["state"], p["state"])}
            ),
            
            "common_knowledge": Primitive(
                "common_knowledge",
                arrow(boolean_type, boolean_type),
                lambda p: {**p, "end": self.common_knowledge(p["state"])}
            ),
            
            "believes": Primitive(
                "believes",
                arrow(agent_type, arrow(boolean_type, boolean_type)),
                lambda a: lambda p: {**a, "end": self.believes(a["state"], p["state"])}
            ),
            
            "confident": Primitive(
                "confident",
                arrow(agent_type, arrow(boolean_type, boolean_type)),
                lambda a: lambda p: {**a, "end": self.confident(a["state"], p["state"])}
            ),
            
            "uncertain": Primitive(
                "uncertain",
                arrow(agent_type, arrow(boolean_type, boolean_type)),
                lambda a: lambda p: {**a, "end": self.uncertain(a["state"], p["state"])}
            ),
            
            "public_announcement": Primitive(
                "public_announcement",
                arrow(boolean_type, boolean_type),
                lambda p: {**p, "end": self.public_announcement(p["state"])}
            ),
            
            "private_announcement": Primitive(
                "private_announcement",
                arrow(agent_type, arrow(boolean_type, boolean_type)),
                lambda a: lambda p: {**a, "end": self.private_announcement(a["state"], p["state"])}
            )
        })

def build_logic_executor(temperature: float = 0.1) -> CentralExecutor:
    """Build logic executor with domain.
    
    Args:
        temperature: Temperature for smooth operations
        
    Returns:
        Initialized line executor instance
    """
    domain = load_domain_string(logic_domain_str, domain_parser)
    executor = CentralExecutor(domain)
    
    # Initialize domain and setup predicates
    logic_domain = LogicDomain(temperature)
    logic_domain.setup_predicates(executor)

    executor.visualize = None #visualize_line_predicates
    return executor

def build_ltl_executor(temperature : float = 0.1):
    domain = load_domain_string(ltl_domain_str, domain_parser)
    executor = CentralExecutor(domain)
    
    ltl_domain = LTLDomain(temperature)
    ltl_domain.setup_predicates(executor)
    
    executor.visualize = None #visualize_line_predicates
    return executor

def build_ltl_executor(temperature : float = 0.1):
    domain = load_domain_string(epistemic_domain_str, domain_parser)
    executor = CentralExecutor(domain)
    
    epistemic_domain = EpistemicDomain(temperature)
    epistemic_domain.setup_predicates(executor)
    
    executor.visualize = None #visualize_line_predicates
    return executor


