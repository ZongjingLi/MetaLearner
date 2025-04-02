import torch
import torch.nn as nn
from nltk.tokenize import RegexpTokenizer
from typing import List, Optional, Union

from helchriss.utils.tokens import *
from helchriss.utils.vocab  import *

class TextEncoder(nn.Module):
    def __init__(
        self, 
        embedding_dim: int = 300,
        vocab_size: Optional[int] = None,
        sequences: Optional[List[str]] = None,
        min_token_count: int = 1,
        delim: str = ' ',
        punct_to_keep: Optional[List[str]] = None,
        punct_to_remove: Optional[List[str]] = None
    ):
        super().__init__()
        """
        Initialize the text encoder with embedding layer.
        
        Args:
            embedding_dim: Dimension of the token embeddings
            vocab_size: Size of vocabulary (if predefined)
            sequences: List of sequences to build vocabulary from
            min_token_count: Minimum count for a token to be included in vocab
            delim: Delimiter for tokenization
            punct_to_keep: Punctuation marks to keep
            punct_to_remove: Punctuation marks to remove
        """
        self.embedding_dim = embedding_dim
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        # Build or initialize vocabulary
        if sequences is not None:
            self.token_to_idx = build_vocab(
                sequences,
                min_token_count=min_token_count,
                delim=delim,
                punct_to_keep=punct_to_keep,
                punct_to_remove=punct_to_remove
            )
            vocab_size = len(self.token_to_idx)
        else:
            assert vocab_size is not None, "Either vocab_size or sequences must be provided"
            self.token_to_idx = SPECIAL_TOKENS.copy()
        
        self.idx_to_token = reverse_diction(self.token_to_idx)
        self.vocab_size = vocab_size
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=SPECIAL_TOKENS['<NULL>'])
        
    def tokenize(
        self, 
        text: str,
        add_start_token: bool = True,
        add_end_token: bool = True
    ) -> List[str]:
        """Tokenize input text."""
        return tokenize(
            text,
            add_start_token=add_start_token,
            add_end_token=add_end_token
        )
    
    def encode_tokens(
        self, 
        tokens: List[str],
        allow_unk: bool = True
    ) -> torch.Tensor:
        """Convert tokens to indices."""
        return torch.tensor(encode(tokens, self.token_to_idx, allow_unk=allow_unk))
    
    def get_embeddings(
        self, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Get embeddings for input indices."""
        return self.embedding(indices)
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        add_start_token: bool = True,
        add_end_token: bool = True,
        allow_unk: bool = True,
        return_tokens: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Encode text to token embeddings.
        
        Args:
            text: Input text or list of texts
            add_start_token: Whether to add start token
            add_end_token: Whether to add end token
            allow_unk: Whether to allow unknown tokens
            return_tokens: Whether to return the tokens along with embeddings
            
        Returns:
            Token embeddings tensor or tuple of (embeddings, tokens) if return_tokens=True
        """
        if isinstance(text, str):
            text = [text]
            
        all_tokens = []
        all_embeddings = []
        
        for t in text:
            # Tokenize
            tokens = self.tokenize(
                t,
                add_start_token=add_start_token,
                add_end_token=add_end_token
            )
            
            # Convert to indices
            indices = self.encode_tokens(tokens, allow_unk=allow_unk)
            
            # Get embeddings
            embeddings = self.get_embeddings(indices)
            
            all_tokens.append(tokens)
            all_embeddings.append(embeddings)
        
        # Stack embeddings
        embeddings_tensor = torch.stack(all_embeddings)
        
        if return_tokens:
            return embeddings_tensor, all_tokens
        return embeddings_tensor
    
    def decode(
        self,
        indices: torch.Tensor,
        delim: Optional[str] = None,
        stop_at_end: bool = True
    ) -> Union[List[str], str]:
        """Decode indices back to tokens/text."""
        if indices.dim() == 2:
            decoded = []
            for seq in indices:
                decoded.append(decode(
                    seq.tolist(),
                    self.idx_to_token,
                    delim=delim,
                    stop_at_end=stop_at_end
                ))
            return decoded
        return decode(
            indices.tolist(),
            self.idx_to_token,
            delim=delim,
            stop_at_end=stop_at_end
        )
        
    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        torch.save({
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'vocab_size': self.vocab_size,
            'embedding_state': self.embedding.state_dict()
        }, path)
        
    def load_vocab(self, path: str):
        """Load vocabulary from file."""
        checkpoint = torch.load(path)
        self.token_to_idx = checkpoint['token_to_idx']
        self.idx_to_token = checkpoint['idx_to_token']
        self.vocab_size = checkpoint['vocab_size']
        self.embedding.load_state_dict(checkpoint['embedding_state'])

# Example usage:
if __name__ == "__main__":
    # Initialize with some example sequences
    sequences = [
        "Hello world!",
        "This is a test sequence.",
        "Another example for vocabulary building."
    ]
    
    # Create encoder
    encoder = TextEncoder(
        embedding_dim=128,
        sequences=sequences,
        punct_to_remove=['.', '!', ',']
    )
    
    # Encode single text
    text = "Hello testing world first, second malganis"
    embeddings, tokens = encoder.encode_text(text, return_tokens=1)
    print(tokens)
    #print(embeddings[0,:])
    #print(f"Shape of embeddings: {embeddings.shape}")
    
    # Encode multiple texts
    #texts = ["First sequence test", "Second sequence"]
    #embeddings, tokens = encoder.encode_text(texts, return_tokens=True)
    #print(f"Shape of batch embeddings: {embeddings.shape}")
    #print(f"Tokens: {tokens}")