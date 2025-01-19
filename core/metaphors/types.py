import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict

class TypeChecker:
    @staticmethod
    def parse_vector_type(type_str: str) -> Optional[Tuple[str, List[str]]]:
        """Parse vector type string like "vector[float,['64']]" into (base_type, dimensions)"""
        if not type_str.startswith("vector["):
            return None
        try:
            content = type_str[7:-1]
            base_type, dims = content.split(',', 1)
            dims = eval(dims)
            return (base_type, dims)
        except:
            return None

    @staticmethod
    def is_type_congruent(type1: str, type2: str) -> bool:
        """Check if two types are congruent (can be mapped between)"""
        if type1 == 'object' or type2 == 'object' or type1 is None or type2 is None:
            return True

        vec1 = TypeChecker.parse_vector_type(type1)
        vec2 = TypeChecker.parse_vector_type(type2)

        if vec1 and vec2:
            base1, dims1 = vec1
            base2, dims2 = vec2
            if base1 != base2:
                return False
            if all(isinstance(d, str) and d.isdigit() for d in dims1) and \
               all(isinstance(d, str) and d.isdigit() for d in dims2):
                return True
            return False

        if not vec1 and not vec2:
            return type1 == type2

        return False

class PredicateConnectionMatrix(nn.Module):
    def __init__(self, source_domain, target_domain):
        super().__init__()
        self.source_predicates = list(source_domain.predicates.keys())
        self.target_predicates = list(target_domain.predicates.keys())
        
        compatible_pairs = []
        connections_count = 0
        
        for s_pred in self.source_predicates:
            s_info = source_domain.predicates[s_pred]
            s_type = s_info['type']
            
            for t_pred in self.target_predicates:
                t_info = target_domain.predicates[t_pred]
                t_type = t_info['type']
                
                if not TypeChecker.is_type_congruent(s_type, t_type):
                    continue
                
                if len(s_info['parameters']) != len(t_info['parameters']):
                    continue
                
                types_compatible = True
                for s_param, t_param in zip(s_info['parameters'], t_info['parameters']):
                    s_param_type = s_param.split('-')[1] if '-' in s_param else None
                    t_param_type = t_param.split('-')[1] if '-' in t_param else None
                    
                    if s_param_type and t_param_type:
                        s_full_type = source_domain.types.get(s_param_type)
                        t_full_type = target_domain.types.get(t_param_type)
                        
                        if not TypeChecker.is_type_congruent(s_full_type, t_full_type):
                            types_compatible = False
                            break
                
                if types_compatible:
                    compatible_pairs.append((s_pred, t_pred))
                    connections_count += 1
        
        self.weight = nn.Parameter(torch.rand(connections_count)) if connections_count > 0 else None
        self.connection_to_idx = {pair: idx for idx, pair in enumerate(compatible_pairs)}

    def get_binary_regularization_loss(self) -> torch.Tensor:
        """Calculate regularization loss to encourage binary weights"""
        if self.weight is None:
            return torch.tensor(0.0)
        
        weights = torch.sigmoid(self.weight)
        # Encourage weights to be either 0 or 1 using binary cross entropy
        reg_loss = -(weights * torch.log(weights + 1e-10) + 
                    (1 - weights) * torch.log(1 - weights + 1e-10))
        return reg_loss.mean()

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the full connection matrix and regularization loss"""
        if self.weight is None:
            return torch.zeros(len(self.source_predicates), len(self.target_predicates)), torch.tensor(0.0)
        
        full_matrix = torch.zeros(len(self.source_predicates), len(self.target_predicates))
        
        for (s_pred, t_pred), idx in self.connection_to_idx.items():
            s_idx = self.source_predicates.index(s_pred)
            t_idx = self.target_predicates.index(t_pred)
            full_matrix[s_idx, t_idx] = torch.sigmoid(self.weight[idx])
        
        reg_loss = self.get_binary_regularization_loss()    
        return full_matrix, reg_loss
        
    def get_connection_weight(self, source_pred: str, target_pred: str) -> torch.Tensor:
        """Get the connection weight between source and target predicates"""
        if self.weight is None:
            return torch.tensor(0.0)
            
        pair = (source_pred, target_pred)
        if pair not in self.connection_to_idx:
            return torch.tensor(0.0)
            
        idx = self.connection_to_idx[pair]
        return torch.sigmoid(self.weight[idx])
        
    def set_connection_weight(self, source_pred: str, target_pred: str, value: float = 1.0):
        """Set a specific connection weight"""
        if self.weight is None:
            return
            
        pair = (source_pred, target_pred)
        if pair in self.connection_to_idx:
            idx = self.connection_to_idx[pair]
            self.weight.data[idx] = torch.logit(torch.tensor(value))

class ActionConnectionMatrix(nn.Module):
    def __init__(self, source_domain, target_domain):
        super().__init__()
        self.source_actions = list(source_domain.actions.keys())
        self.target_actions = list(target_domain.actions.keys())
        
        compatible_pairs = []
        connections_count = 0
        
        for s_action in self.source_actions:
            s_info = source_domain.actions[s_action]
            for t_action in self.target_actions:
                t_info = target_domain.actions[t_action]
                
                if len(s_info.parameters) != len(t_info.parameters):
                    continue
                    
                compatible_pairs.append((s_action, t_action))
                connections_count += 1
        
        self.weight = nn.Parameter(torch.rand(connections_count)) if connections_count > 0 else None
        self.connection_to_idx = {pair: idx for idx, pair in enumerate(compatible_pairs)}

    def get_binary_regularization_loss(self) -> torch.Tensor:
        """Calculate regularization loss to encourage binary weights"""
        if self.weight is None:
            return torch.tensor(0.0)
        
        weights = torch.sigmoid(self.weight)
        # Encourage weights to be either 0 or 1 using binary cross entropy
        reg_loss = -(weights * torch.log(weights + 1e-10) + 
                    (1 - weights) * torch.log(1 - weights + 1e-10))
        return reg_loss.mean()

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the full connection matrix and regularization loss"""
        if self.weight is None:
            return torch.zeros(len(self.source_actions), len(self.target_actions)), torch.tensor(0.0)
            
        full_matrix = torch.zeros(len(self.source_actions), len(self.target_actions))
        
        for (s_action, t_action), idx in self.connection_to_idx.items():
            s_idx = self.source_actions.index(s_action)
            t_idx = self.target_actions.index(t_action)
            full_matrix[s_idx, t_idx] = torch.sigmoid(self.weight[idx])
            
        reg_loss = self.get_binary_regularization_loss()
        return full_matrix, reg_loss
        
    def get_connection_weight(self, source_action: str, target_action: str) -> torch.Tensor:
        """Get the connection weight between source and target actions"""
        if self.weight is None:
            return torch.tensor(0.0)
            
        pair = (source_action, target_action)
        if pair not in self.connection_to_idx:
            return torch.tensor(0.0)
            
        idx = self.connection_to_idx[pair]
        return torch.sigmoid(self.weight[idx])
        
    def set_connection_weight(self, source_action: str, target_action: str, value: float = 1.0):
        """Set a specific connection weight"""
        if self.weight is None:
            return
            
        pair = (source_action, target_action)
        if pair in self.connection_to_idx:
            idx = self.connection_to_idx[pair]
            self.weight.data[idx] = torch.logit(torch.tensor(value))