import os 
import torch
import torch.nn as nn
import torch.nn.functional as f

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class DeepfusionConfig:
    vocab_size : int = 8192
    embedding_dim : int = 768
    hidden_dim: int = 768
    sliding_attn_size: int = 128 # = -1 for global attn
    num_attn_layers = 12
    # "[UNK]", "[START_ID]", "[END_ID]", "[EOT]", "[MASK]", "[BOS]", "[EOS]"

    ratio_global_window = 3
    pad_token : int = 0
    layer_norm_eps: float = 1e-5
    num_heads: int = 12
    rope_base: int = 10_000
    max_seq_len: int = 1024 # Maximum context length

# region RoPE 
# 1. Helper function to rotate half the dimensions
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# 2. Function to apply the rotations to Q and K
def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies RoPE to Q and K tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 3. The RoPE Module
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        # Calculate inverse frequencies: 1 / (base ^ (2i / dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # Use register_buffer so it's moved to the correct device alongside the model
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        # Create a tensor of positions: [0, 1, 2, ..., seq_len - 1]
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Outer product to get frequencies for each position
        freqs = torch.outer(t, self.inv_freq)
        
        # Duplicate freqs to match the head dimension (pairing them up)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Add broadcasting dimensions for batch and num_heads: (1, seq_len, 1, head_dim)
        cos = emb.cos().unsqueeze(0).unsqueeze(2)
        sin = emb.sin().unsqueeze(0).unsqueeze(2)
        return cos, sin
#endregion 

# region Attention
class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, config: DeepfusionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.window_size = config.sliding_attn_size
        
        # Pre-norm
        self.pre_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps, elementwise_affine=True)
        
        # RoPE initialized with the dimension of a SINGLE head
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, base=config.rope_base)
        
        # Linear projections for Attention
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Out projections 
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(p=0.1)
        
    def sliding_window_mask(self, seq_len, device="cpu"):
        if self.window_size == -1:
            return None
            
        idx = torch.arange(seq_len, device=device)

        # pairwise distance
        dist = idx[None, :] - idx[:, None]

        # For Masked Language Modeling, we need BIDIRECTIONAL attention.
        # PyTorch SDPA expects True for elements allowed to participate.
        # We allow all tokens within the sliding window (both past and future).
        mask = torch.abs(dist) <= self.window_size

        return mask

    def forward(self, x:torch.Tensor, pre_norm = True, attention_mask = None):
        batch_size, seq_len, _ = x.shape

        if pre_norm: 
            x = self.pre_norm(x)

        # Project to Q, K, V and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Get RoPE frequencies for the current sequence length
        cos, sin = self.rope(seq_len, x.device)
        
        # Apply RoPE strictly to Q and K
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for SDPA: (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Combine sliding window mask and optional attention_mask
        mask = self.sliding_window_mask(seq_len, x.device)
        
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len), where 1/True is valid token, 0/False is padding
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2) # (batch, 1, 1, seq_len)
            if mask is not None:
                mask = mask.unsqueeze(0).unsqueeze(0) & pad_mask # (1, 1, seq_len, seq_len) & (batch, 1, 1, seq_len)
            else:
                mask = pad_mask
        else:
            if mask is not None:
                mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)
        
        # SDPA returns (batch, heads, seq_len, head_dim)
        attention = f.scaled_dot_product_attention(
            q, k, v,
            attn_mask = mask
        )

        # Transpose back and reshape: (batch, seq_len, hidden_dim)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, seq_len, -1)
        
        out = self.out_proj(attention)
        out = self.dropout(out)

        return out

class DeepfusionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.middle_size = config.hidden_dim * 3 // 2
        assert config.hidden_dim % 2 == 0, "Hidden dim should be divisible for 2"

        self.in_proj = nn.Linear(config.hidden_dim, config.hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.middle_size, config.hidden_dim, bias=False)

    def forward(self, x):
        # batch_size , seq_len , _ = x.shape
        x = self.in_proj(x)
        a, b = x.chunk(2, dim=-1)
        x = f.gelu(a) * b
        x = self.out_proj(x)
        return x

class DeepfusionAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.multi_head_attention = MultiheadAttentionWithRoPE(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_dim, eps = config.layer_norm_eps, elementwise_affine=True)
        self.mlp = DeepfusionMLP(config)

    def forward(self, x, pre_norm=True, attention_mask = None):
        # Attention with residual connection
        # (MultiheadAttentionWithRoPE handles its own pre_norm internally)
        x = x + self.multi_head_attention(x, pre_norm=pre_norm, attention_mask=attention_mask)
        
        # MLP with residual connection
        x = x + self.mlp(self.mlp_norm(x))
        return x

# endregion

# region Embedding

class DeepfusionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token
        )
        self.layer_norm = nn.LayerNorm(
            config.embedding_dim, 
            eps=config.layer_norm_eps,
            elementwise_affine=True
        )

    def forward(self, idx: torch.Tensor):
        x = self.token_embedding(idx)
        x = self.layer_norm(x)
        return x

# endregion

# region Decoder
class DeepfusionDecoderHead(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps = config.layer_norm_eps, elementwise_affine=True)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(config.hidden_dim ,config.vocab_size, bias=True)

    def forward(self, x): 
        x = self.dense(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x
# endregion

class DeepfusionLM(nn.Module):

    def __init__(self, config):
        super(DeepfusionLM, self).__init__()
        self.config = config
        self.embedding = DeepfusionEmbedding(config)
        self.attention = DeepfusionAttentionLayer(config)
        self.attentions = nn.ModuleList()
        
        orig_window_size = config.sliding_attn_size
        for i in range(1, config.num_attn_layers):        
            # if i % config.ratio_global_window == 0:
            config.sliding_attn_size = -1 # All layers are full attention 
            config.rope_base = 160_000
            # else:
            #     config.sliding_attn_size = orig_window_size
            #     config.rope_base = 10_000
            
            self.attentions.append(DeepfusionAttentionLayer(config))
        
        # Restore original config state
        config.sliding_attn_size = orig_window_size
        self.lm_head = DeepfusionDecoderHead(config)
        self.apply(_init_weights)

    def forward(self, input_ids, attention_mask = None):
        x = self.embedding(input_ids)
        x = self.attention(x, pre_norm=False, attention_mask = attention_mask)
        for layer in self.attentions:
            x = layer(x, attention_mask = attention_mask)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def fill_mask(self, idx, mask_token_id, temperature=1.0, top_k=None, sample=False):
        """
        Predicts and fills [MASK] tokens in the input sequence.
        - mask_token_id: The ID of the [MASK] token in your vocabulary.
        - sample: If True, samples from the distribution. If False, takes the argmax (greedy).
        """
        self.eval()
        
        # Ensure the input doesn't exceed the max context length
        idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, :self.config.max_seq_len]
        
        # Forward pass to get logits for all tokens
        logits = self(idx_cond)
        
        # Identify where the mask tokens are in the sequence
        mask_positions = (idx_cond == mask_token_id)
        
        # If no mask tokens are found, return original sequence
        if not mask_positions.any():
            return idx_cond
            
        # Get logits specifically for the masked positions
        # shape: (num_masked_tokens, vocab_size)
        mask_logits = logits[mask_positions] / temperature
        
        if sample:
            # Optional: Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(mask_logits, min(top_k, mask_logits.size(-1)))
                mask_logits[mask_logits < v[:, [-1]]] = -float('Inf')
            
            # Convert to probabilities and sample
            probs = f.softmax(mask_logits, dim=-1)
            predicted_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            # Greedy prediction (standard for MLM)
            predicted_tokens = mask_logits.argmax(dim=-1)
            
        # Create a copy and fill in the predictions
        filled_idx = idx_cond.clone()
        filled_idx[mask_positions] = predicted_tokens
        
        return filled_idx
        
def _init_weights(module):
    """
    Standard initialization for Transformer models:
    - Linear/Embedding weights: Normal(0, 0.02)
    - LayerNorm: weight=1, bias=0
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)

if __name__ == "__main__": 
    CONFIG = DeepfusionConfig()

    model = DeepfusionLM(CONFIG)
    
    test_seq = torch.randint(0, CONFIG.vocab_size, (1, 128))
    output = model(test_seq)
    
    print(f"Input shape: {test_seq.shape}")
    print(f"Output shape: {output.shape}")
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate VRAM taken by parameters (in MB)
    # Each parameter takes: num_elements * size_of_element (e.g., 4 bytes for float32)
    vram_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    vram_mb = vram_bytes / (1024 ** 2)
    
    print(f"Number of parameters: {num_params:,}")
    print(f"Number of trainable parameters: {num_trainable_params:,}")
    print(f"Inference VRAM (parameters only): {vram_mb:.2f} MB")
    
    # Training VRAM Estimation (assuming AdamW optimizer)
    # float32: 4 (W) + 4 (G) + 8 (Opt) = 16 bytes/param
    # bfloat16 mixed: 2 (W) + 2 (G) + 8 (Opt) + 4 (Master W) = 16 bytes/param
    # bfloat16 pure: 2 (W) + 2 (G) + 4 (Opt) = 8 bytes/param
    
    fp32_train_mb = (num_params * 16) / (1024 ** 2)
    bf16_mixed_mb = (num_params * 16) / (1024 ** 2) 
    bf16_pure_mb = (num_params * 8) / (1024 ** 2)

    print("-" * 30)
    print(f"ESTIMATED TRAINING VRAM (Static):")
    print(f"Float32 Training:   ~{fp32_train_mb:.2f} MB")
    print(f"Bfloat16 Mixed:     ~{bf16_mixed_mb:.2f} MB")
    print(f"Bfloat16 Pure:      ~{bf16_pure_mb:.2f} MB")
    print("-" * 30)
    print("Note: Activations (forward pass memory) are not included and will")
    print("increase with batch size and sequence length.")