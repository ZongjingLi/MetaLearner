import re
import torch
import json
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer

@dataclass
class LeanHypothesis:
    """Represents a hypothesis in a Lean 4 proof state"""
    id: str
    type_expr: str
    is_local_def: bool = False
    name: Optional[str] = None
    value: Optional[str] = None

@dataclass
class LeanGoal:
    """Represents a goal in a Lean 4 proof state"""
    mvarId: str
    userName: Optional[str]
    type: str
    context: List[LeanHypothesis]

@dataclass
class LeanProofState:
    """Represents the full Lean 4 proof state"""
    goals: List[LeanGoal]
    mainGoal: Optional[str] = None
    remainingGoals: int = 0

class Lean4ProofStateEncoder:
    """Encoder for Lean 4 proof states"""
    
    def __init__(
        self,
        tokenizer_name: str = "EleutherAI/gpt-neox-20b",
        max_hypothesis_tokens: int = 128,
        max_goal_tokens: int = 256,
        max_total_tokens: int = 1024,
        special_tokens: bool = True
    ):
        """
        Initialize the encoder with tokenizer and configuration.
        
        Args:
            tokenizer_name: Name of the HuggingFace tokenizer to use
            max_hypothesis_tokens: Maximum tokens for each hypothesis
            max_goal_tokens: Maximum tokens for each goal
            max_total_tokens: Maximum total tokens for the entire proof state
            special_tokens: Whether to add special tokens for proof state components
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_hypothesis_tokens = max_hypothesis_tokens
        self.max_goal_tokens = max_goal_tokens
        self.max_total_tokens = max_total_tokens
        
        # Set padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add a padding token if neither pad_token nor eos_token exists
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Add special tokens for proof state components
        if special_tokens:
            special_tokens_dict = {
                'additional_special_tokens': [
                    '[GOAL]', '[/GOAL]',
                    '[HYP]', '[/HYP]',
                    '[NAME]', '[/NAME]',
                    '[TYPE]', '[/TYPE]',
                    '[VALUE]', '[/VALUE]',
                    '[CONTEXT]', '[/CONTEXT]'
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
    
    def parse_proof_state(self, proof_state_text: str) -> LeanProofState:
        """
        Parse a text representation of a Lean 4 proof state.
        
        Args:
            proof_state_text: Text representation of the proof state
            
        Returns:
            Structured LeanProofState object
        """
        # This is a simplified parser - a real implementation would need
        # to handle all the nuances of Lean 4's proof state format
        goals = []
        
        # Split into goals
        goal_texts = re.split(r'(?m)^case\s+|^goal\s+', proof_state_text)
        
        for goal_text in goal_texts:
            if not goal_text.strip():
                continue
                
            # Extract goal ID and name
            mvar_match = re.search(r'⊢\s*(.*?)$', goal_text, re.MULTILINE)
            if not mvar_match:
                continue
                
            goal_type = mvar_match.group(1).strip()
            
            # Extract hypotheses
            hyp_lines = re.findall(r'(?m)^(.*?)\s*:\s*(.*?)$', goal_text)
            
            hypotheses = []
            for name, type_expr in hyp_lines:
                name = name.strip()
                type_expr = type_expr.strip()
                
                # Check if it's a local definition
                is_local_def = " := " in type_expr
                value = None
                
                if is_local_def:
                    type_parts = type_expr.split(" := ", 1)
                    type_expr = type_parts[0].strip()
                    value = type_parts[1].strip() if len(type_parts) > 1 else None
                
                hyp = LeanHypothesis(
                    id=f"h_{len(hypotheses)}",
                    name=name,
                    type_expr=type_expr,
                    is_local_def=is_local_def,
                    value=value
                )
                hypotheses.append(hyp)
            
            goal = LeanGoal(
                mvarId=f"goal_{len(goals)}",
                userName=None,  # Would extract from case name in full implementation
                type=goal_type,
                context=hypotheses
            )
            goals.append(goal)
        
        return LeanProofState(
            goals=goals,
            remainingGoals=len(goals)
        )
    
    def format_hypothesis(self, hyp: LeanHypothesis, use_special_tokens: bool = True) -> str:
        """Format a hypothesis for encoding"""
        if use_special_tokens:
            if hyp.name:
                result = f"[HYP][NAME]{hyp.name}[/NAME][TYPE]{hyp.type_expr}[/TYPE]"
            else:
                result = f"[HYP][TYPE]{hyp.type_expr}[/TYPE]"
                
            if hyp.is_local_def and hyp.value:
                result += f"[VALUE]{hyp.value}[/VALUE]"
                
            result += "[/HYP]"
            return result
        else:
            if hyp.is_local_def and hyp.value:
                return f"{hyp.name} : {hyp.type_expr} := {hyp.value}"
            else:
                return f"{hyp.name} : {hyp.type_expr}"
    
    def format_goal(self, goal: LeanGoal, use_special_tokens: bool = True) -> str:
        """Format a goal for encoding"""
        if use_special_tokens:
            context = "[CONTEXT]" + "".join(
                self.format_hypothesis(hyp, use_special_tokens)
                for hyp in goal.context
            ) + "[/CONTEXT]"
            
            return f"[GOAL]{context}[TYPE]{goal.type}[/TYPE][/GOAL]"
        else:
            context = "\n".join(
                self.format_hypothesis(hyp, use_special_tokens)
                for hyp in goal.context
            )
            
            return f"{context}\n⊢ {goal.type}"
    
    def encode_proof_state(
        self, 
        proof_state: LeanProofState,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        use_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a Lean 4 proof state into tokenized format.
        
        Args:
            proof_state: The proof state to encode
            return_tensors: Format of returned tensors ('pt' for PyTorch)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            use_special_tokens: Whether to use special tokens
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        # Format the proof state
        formatted_text = ""
        for goal in proof_state.goals:
            formatted_text += self.format_goal(goal, use_special_tokens) + "\n"
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_text,
            padding=padding,
            truncation=truncation,
            max_length=self.max_total_tokens,
            return_tensors=return_tensors
        )
        
        return encoded
    
    def encode_proof_state_hierarchical(
        self, 
        proof_state: LeanProofState,
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        """
        Encode a proof state hierarchically, preserving structure.
        
        Args:
            proof_state: The proof state to encode
            return_tensors: Format of returned tensors
            
        Returns:
            Dictionary with hierarchical encoding of goals and hypotheses
        """
        result = {
            "goals": [],
            "num_goals": len(proof_state.goals)
        }
        
        for goal in proof_state.goals:
            goal_type_encoding = self.tokenizer(
                goal.type, 
                truncation=True,
                max_length=self.max_goal_tokens,
                return_tensors=return_tensors
            )
            
            hypotheses_encodings = []
            for hyp in goal.context:
                hyp_text = self.format_hypothesis(hyp, use_special_tokens=False)
                hyp_encoding = self.tokenizer(
                    hyp_text,
                    truncation=True,
                    max_length=self.max_hypothesis_tokens,
                    return_tensors=return_tensors
                )
                hypotheses_encodings.append({
                    "encoding": hyp_encoding,
                    "is_def": hyp.is_local_def,
                    "name": hyp.name
                })
            
            goal_encoding = {
                "type_encoding": goal_type_encoding,
                "hypotheses": hypotheses_encodings,
                "num_hypotheses": len(goal.context)
            }
            
            result["goals"].append(goal_encoding)
        
        return result

    def batch_encode_proof_states(
        self,
        proof_states: List[LeanProofState],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of proof states.
        
        Args:
            proof_states: List of proof states to encode
            return_tensors: Format of returned tensors
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            Dictionary with batched encodings
        """
        formatted_texts = []
        for proof_state in proof_states:
            text = ""
            for goal in proof_state.goals:
                text += self.format_goal(goal) + "\n"
            formatted_texts.append(text)
        
        # Tokenize the batch
        return self.tokenizer(
            formatted_texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_total_tokens,
            return_tensors=return_tensors
        )

    def decode_to_proof_state(self, tokens: torch.Tensor) -> str:
        """
        Decode tokens back to a proof state representation.
        This is useful for debugging or generating proof states.
        
        Args:
            tokens: Tensor of token IDs
            
        Returns:
            String representation of the proof state
        """
        return self.tokenizer.decode(tokens)

# Example usage
def example():
    # Example Lean 4 proof state
    proof_state_text = """
    case induction_step
    x y : Nat
    h : x = y → y = x
    ⊢ x + 1 = y + 1 → y + 1 = x + 1
    
    case base_case
    x : Nat
    ⊢ x = x
    """
    
    encoder = Lean4ProofStateEncoder()
    
    # Parse the proof state
    proof_state = encoder.parse_proof_state(proof_state_text)
    
    # Encode the proof state
    encoding = encoder.encode_proof_state(proof_state)
    
    # Print encoding shape
    print(f"Encoding shape: {encoding['input_ids'].shape}")
    
    # Hierarchical encoding
    hier_encoding = encoder.encode_proof_state_hierarchical(proof_state)
    
    # Print hierarchical encoding structure
    print(f"Number of goals: {hier_encoding['num_goals']}")
    for i, goal in enumerate(hier_encoding['goals']):
        print(f"Goal {i}: {goal['num_hypotheses']} hypotheses")
    
    # Decode back to text (for debugging)
    decoded = encoder.decode_to_proof_state(encoding['input_ids'][0])
    print(f"Decoded proof state (truncated):\n{decoded[:200]}...")

if __name__ == "__main__":
    example()