"""
Implementation of a LLaDA-inspired Masked Diffusion Model for Text
using PURE BYTE-LEVEL TOKENIZATION and Mixed Precision Training (AMP).
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import random
import math
import time
from tqdm import tqdm
from contextlib import nullcontext, suppress
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter #tensorboard --logdir=runs

#For more predictable results.
torch.manual_seed(0)
random.seed(0)

# --- Configuration ---
class CONFIG:
    """ Stores configuration parameters for the model, training, and data """

    # --- MLA Test --- 
    head_dim = 32
    v_head_dim = head_dim
    nope_head_dim = 32
    rope_head_dim = 64

    kv_lora_rank = 64
    q_lora_rank = 3 * kv_lora_rank

    # --- Special Byte-Level Token IDs ---
    PAD_ID: int = 256
    MASK_ID: int = 257
    UNK_ID: int = 258
    NUM_SPECIAL_TOKENS: int = 3

    # --- Data Files & Dirs ---
    train_data_file: str = "./train.txt"
    validation_data_file: str = "./validation.txt"
    checkpoint_dir: Path = Path("./llada_byte_checkpoints_amp_v4")
    final_model_dir: Path = Path("./llada_byte_model_final_amp_v4")
    name = "Diffusion-LM" 

    # --- Model Architecture ---
    vocab_size: int = 256 + NUM_SPECIAL_TOKENS
    model_dim: int = 512 #Higher = better memorization (good range is 128-2048~)
    num_layers: int = 8 #Higher = better logic but harder to train (good range is 4-32~)
    num_heads: int = 16
    kv_latent_dim: int = model_dim // 4 # Example: 1024 // 4 = 256
    dropout = 0

    assert kv_latent_dim > 0 and kv_latent_dim <= model_dim, "kv_latent_dim must be positive and <= model_dim"

    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256 #Ensure it aligns with model_dim
    norm_eps: float = 1e-5
    use_bias: bool = False # Bias in Attention/FFN layers

    # --- RoPE Specific ---
    rope_theta: float = 10000.0
    rope_max_length_buffer: int = 2048 # Max length for RoPE precomputation buffer - should be >= max(train_max_len, max_infer_len)

    # --- Training ---
    max_length: int = 768 # Max length for *training* chunks (in bytes), this increases model requirements.
    batch_size: int = 8 #Lower if your computer suffers
    learning_rate: float = 2e-4 
    num_epochs: int = 500 # Train for this many epochs
    logging_steps: int = 250 # Log loss every N steps
    predict_steps: int = 2500 # Run validation inference every N steps
    gradient_accumulation_steps: int = 2
    gradient_clip_val: float = 1.0 # Max norm for gradient clipping
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4 # Set num_workers=0 to debug DataLoader issues causing None items in collate_fn

    # --- AMP (Mixed Precision) ---
    use_amp: bool = torch.cuda.is_available()

    # Use bfloat16 if available, otherwise float16 for AMP
    amp_dtype: torch.dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    # --- Diffusion ---
    mask_token_id: int = MASK_ID
    pad_token_id: int = PAD_ID
    min_mask_prob: float = 0.075 # Min 't' during training
    loss_t_clamp_min: float = 1e-6 # Min 't' for loss division

    # --- Validation ---
    inference_steps: int = 64 # Number of diffusion steps for validation samples
    num_validation_samples: int = 5 # Number of validation examples to genearte
    
    #  --- Random length ranges for validation inference ---
    val_prompt_min_len: int = 256
    val_prompt_max_len: int = 768
    val_response_min_len: int = 128
    val_response_max_len: int = 256

    verbose_inference: bool = True # Show step-by-step denoising?


# --- Tokenizer Info (Byte Handling) ---
print("--- Using Byte-Level Tokenization ---")
print(f"Byte range: 0-255")
print(f"Special Tokens: PAD={CONFIG.PAD_ID}, MASK={CONFIG.MASK_ID}, UNK={CONFIG.UNK_ID}")
print(f"Effective Vocab Size: {CONFIG.vocab_size}")
if CONFIG.MASK_ID is None or CONFIG.PAD_ID is None:
    raise ValueError("Essential special tokens (MASK_ID, PAD_ID) are not defined.")


# --- Data Loading ---
class ByteLevelTextDataset(Dataset):
    """
    Loads text data from a file as raw bytes, converts bytes to their integer
    representations (0-255), and chunks these integers into fixed-length sequences.
    """
    def __init__(
        self, file_path: Union[str, Path], max_length: int, file_desc: str = "Training"
    ):
        self.max_length = max_length
        self.examples: List[List[int]] = []
        self.file_path = Path(file_path)
        self.file_desc = file_desc
        print(f"Reading {self.file_desc} data as BYTES from '{self.file_path}'...")

        byte_sequence = self._read_bytes()
        if not byte_sequence:
            print(f"Warning: {self.file_desc} file '{self.file_path}' empty/unreadable.")
            return

        self._process_bytes(byte_sequence)

    def _read_bytes(self) -> bytes:
        try:
            return self.file_path.read_bytes()
        except FileNotFoundError:
            print("FileNotFoundError")
            if self.file_desc == "Training":
                self._create_dummy_byte_file()
                with suppress(FileNotFoundError):
                    return self.file_path.read_bytes()
            return b""
        except Exception as e:
            print(f"Error reading file '{self.file_path}': {e}")
            return b""

    def _create_dummy_byte_file(self):
        #Creates a dummy data file if the original is missing.
        print(f"Creating dummy byte data file: {self.file_path}")
        dummy_text = ("Dummy sentence for training.\nIncludes newlines.\n" * 20)
        dummy_bytes = dummy_text.encode('utf-8', errors='replace')
        try:
            self.file_path.write_bytes(dummy_bytes * 100)
        except IOError as e:
            print(f"Error creating dummy byte file '{self.file_path}': {e}")

    def _process_bytes(self, byte_sequence: bytes):
        #Converts byte sequence to integer list and chunks into examples.
        all_byte_ids = list(byte_sequence)
        print(f"Total bytes read in {self.file_desc}: {len(all_byte_ids)}")
        print(f"Chunking {self.file_desc} data into sequences of length {self.max_length}...")

        self.examples = [
            all_byte_ids[i : i + self.max_length]
            for i in range(0, len(all_byte_ids) - self.max_length + 1, self.max_length)
        ]

        if not self.examples:
            print(f"WARNING: No sequences created for {self.file_desc}. "
                f"Content length {len(all_byte_ids)} < max_length {self.max_length}?")
        else:
            print(f"Created {len(self.examples)} sequences for {self.file_desc}.")

    def __len__(self) -> int:
        #Returns the number of sequences.
        return len(self.examples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        #Returns the byte sequence (as tensor) at the index, or None if invalid.
        if not 0 <= idx < len(self.examples):
            # Handles potential issues with multiprocessing/DataLoader workers
            print(f"Warning: Invalid index {idx} requested from {self.file_desc} dataset of size {len(self.examples)}")
            return None
        return {"input_ids": torch.tensor(self.examples[idx], dtype=torch.long)}


def collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of byte sequences, handling potential None items.
    Pads sequences and creates an attention mask.
    """
    valid_items = [item for item in batch if item is not None and 'input_ids' in item]

    if not valid_items:
        print("Warning: Collate function received empty or all-None/invalid batch.")
        return {"input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long)}

    input_ids = [item['input_ids'] for item in valid_items]
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=CONFIG.pad_token_id
    )
    attention_mask = (padded_input_ids != CONFIG.pad_token_id).long()
    return {"input_ids": padded_input_ids, "attention_mask": attention_mask}


def load_validation_sequences(
    file_path: Union[str, Path], max_length: int, num_samples: int
) -> List[Dict[str, torch.Tensor]]:
    #Loads the first `num_samples` valid byte sequences from the validation file.
    print(f"--- Loading Validation Byte Sequences ({num_samples} samples) ---")
    validation_sequences = []
    try:
        val_dataset = ByteLevelTextDataset(file_path, max_length, file_desc="Validation")
        num_to_load = min(num_samples, len(val_dataset))
        if num_to_load == 0: 
            print("No validation sequences available.")
            return []

        for i in range(num_to_load):
            sample = val_dataset[i]
            if sample is None: continue
            ids = sample['input_ids']
            mask = (ids != CONFIG.pad_token_id).long().unsqueeze(0)
            ids = ids.unsqueeze(0)
            validation_sequences.append({"input_ids": ids, "attention_mask": mask})
        print(f"Successfully loaded {len(validation_sequences)} validation byte sequences.")
    except Exception as e: 
        print(f"Error loading validation sequences: {e}")
        validation_sequences = []

    return validation_sequences


# --- Model Architecture Components ---

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-6, bias=False):
        """
        Root Mean Square Layer Normalization
        
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """ Precomputes the complex frequency tensor for RoPE. """

    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.    

    Based on: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ Reshapes RoPE frequencies for broadcasting with attention tensors (bs, n_heads, seq_len, dim). """

    """
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    
    From: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    ndim = x.ndim
    assert ndim == 4
    assert freqs_cis.shape == (x.shape[2], x.shape[3]), f"RoPE shape mismatch: freqs {freqs_cis.shape}, x expects ({x.shape[2]}, {x.shape[3]})"
    shape = [1, 1, x.shape[2], x.shape[3]]
    return freqs_cis.view(shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Applies RoPE to query and key tensors (shape: bs, n_heads, seq_len, head_dim). """

    """
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        Apply rotary embeddings to input tensors using the given frequency tensor.

        This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
        frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
        returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
            
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        From: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    freqs_cis = freqs_cis.to(xq_c.device)
    freqs_cis_reshaped = reshape_for_broadcast(freqs_cis, xq_c)

    xq_out = torch.view_as_real(xq_c * freqs_cis_reshaped).flatten(3)
    xk_out = torch.view_as_real(xk_c * freqs_cis_reshaped).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- MLA (NOT USED RIGHT NOW) ---
class MultiHeadLatentAttention(nn.Module):
    """
    Multi Head Latent Attention 
    paper: https://arxiv.org/pdf/2405.04434
    
    TLDR: 
    kv are low ranks, this verient of attention project q,k,v to low rank to save memory,
    replace linear with lora(ish) layers

    source: https://github.com/joey00072/Multi-Head-Latent-Attention-MLA-
    """
    def __init__(self, config: CONFIG):
        super().__init__()
        
        
        assert config.v_head_dim is not None , f"v_head_dim is not defined {config.v_head_dim=}"
        assert config.q_lora_rank is not None , f"q_lora_rank is not defined {config.q_lora_rank=}"
        assert config.kv_lora_rank is not None , f"kv_lora_rank is not defined {config.kv_lora_rank=}"
        assert config.rope_head_dim is not None , f"rope_head_dim is not defined {config.rope_head_dim=}"
        
        self.config = config
        
        self.dim = config.model_dim
        self.num_heads = config.num_heads
        self.v_head_dim = config.v_head_dim
        
        self.nope_head_dim = config.nope_head_dim
        self.rope_head_dim = config.rope_head_dim
        
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        
        self.dropout = config.dropout
        
        self.value_dim = self.num_heads * self.v_head_dim
        
        self.nope_dim = self.num_heads * self.nope_head_dim
        self.rope_dim = self.num_heads * self.rope_head_dim  
        
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=self.config.use_bias) 
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=self.config.use_bias)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=self.config.use_bias)
        self.q_norm = RMSNorm(self.q_lora_rank)
        
        
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=self.config.use_bias) 
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=self.config.use_bias)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=self.config.use_bias)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        
        
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim, bias=self.config.use_bias)
        # self.rope_norm = RMSNorm(self.rope_dim) # not in deepseekv2

        self.proj = nn.Linear(self.value_dim , self.dim, bias=self.config.use_bias)
        self.res_dropout = nn.Dropout(p=config.dropout)
        
        
    def forward(self, x: Tensor,mask: torch.Tensor, freqs_cis: Tensor):
        batch_size, seq_len, _ = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope:Tensor = self.decompress_q_nope(norm_q)
        query_rope:Tensor = self.decompress_q_rope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope: Tensor = self.decompress_k_nope(norm_kv)
        value: Tensor = self.decompress_v_linear(norm_kv)
        
        key_rope:Tensor = self.k_rope_linear(x)

        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
        
        key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1,2)
        
        # *** the line that fixes MLA :) ***
        key_rope = key_rope/self.num_heads 

        q_rope,k_rope = apply_rotary_emb(query_rope, key_rope, freqs_cis)
        
        q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        
        q_recombined[:,:,:,:self.nope_head_dim] = query_nope
        q_recombined[:,:,:,self.nope_head_dim:] = q_rope
        
        k_recombined[:,:,:,:self.nope_head_dim] = key_nope
        k_recombined[:,:,:,self.nope_head_dim:] = k_rope

        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True, dropout_p=self.dropout)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)

        output = self.proj(output)
        output = self.res_dropout(output)
        return output
    
class SimplifiedMLAttention(nn.Module):
    """
    Multi-Head Latent Attention (Simplified Version).
    Uses low-rank factorization for K and V projections.
    Applies existing RoPE implementation to decompressed Q and K.
    Does NOT implement DeepSeek's full Decoupled RoPE.

    https://github.com/joey00072/Multi-Head-Latent-Attention-MLA-/blob/master/mla.py
    """
    def __init__(self, config: CONFIG):
        super().__init__()
        self.n_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.kv_latent_dim = config.kv_latent_dim 
        assert self.head_dim * self.n_heads == config.model_dim, "model_dim must be divisible by num_heads"

        self.wq = nn.Linear(config.model_dim, config.num_heads * self.head_dim, bias=config.use_bias)
        
        self.w_kv_compress = nn.Linear(config.model_dim, self.kv_latent_dim, bias=config.use_bias)

        self.w_k_decompress = nn.Linear(self.kv_latent_dim, config.num_heads * self.head_dim, bias=config.use_bias)
        self.w_v_decompress = nn.Linear(self.kv_latent_dim, config.num_heads * self.head_dim, bias=config.use_bias)

        self.wo = nn.Linear(config.num_heads * self.head_dim, config.model_dim, bias=config.use_bias)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        c_kv = self.w_kv_compress(x)

        xk_full = self.w_k_decompress(c_kv)
        xv_full = self.w_v_decompress(c_kv)

        xk = xk_full.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv_full.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        current_freqs_cis = freqs_cis[:seqlen] 
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=current_freqs_cis[:, :self.head_dim//2])

        attn_mask_for_pytorch = None
        if mask is not None:
            assert mask.ndim == 2, f"Padding mask should have ndim=2, got {mask.ndim}"
            attn_mask_for_pytorch = (1.0 - mask[:, None, None, :].float()) * torch.finfo(xq.dtype).min

        # 7. Scaled Dot-Product Attention (using RoPE'd Q/K and decompressed V)
        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attn_mask_for_pytorch,
            is_causal=False # Keep bidirectional for LLaDA https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_output)

class FeedForward(nn.Module):
    """Modified SwiGLU Feed-Forward Network. """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 256, use_bias: bool = False):
        super().__init__()
        if hidden_dim is None: 
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=use_bias)

        #https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
    def forward(self, x): 
        return self.w2(F.mish(self.w1(x), inplace=True) * self.w3(x)) #Silu -> Mish. Slower but better/more consistent results. I suggest Mish.
    
class TransformerBlock(nn.Module):
    """ Simple Transformer block: Attention and FFN with pre-normalization. """
    def __init__(self, config: CONFIG):
        super().__init__()

        self.attention = SimplifiedMLAttention(config)

        self.feed_forward = FeedForward(dim=config.model_dim, multiple_of=config.multiple_of, use_bias=config.use_bias)
        self.attention_norm = RMSNorm(config.model_dim, eps=config.norm_eps) 
        self.ffn_norm = RMSNorm(config.model_dim, eps=config.norm_eps)      

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, padding_mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
        
# --- Main Model ---
class MaskPredictorLLaDA(nn.Module):
    """ LLaDA-inspired Mask Predictor Model (Byte-Level Vocab). """
    def __init__(self, config: CONFIG):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.model_dim, eps=config.norm_eps)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=config.use_bias)
        self.register_buffer("freqs_cis", precompute_freqs_cis(config.model_dim // config.num_heads, config.rope_max_length_buffer, theta=config.rope_theta), persistent=False)
        self.apply(self._init_weights)
        print(f"--- Byte-Level Model Initialized ---\nVocab Size: {config.vocab_size}\nLayers: {config.num_layers}, Dim: {config.model_dim}, Heads: {config.num_heads}\nParams: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}\nRoPE Buffer Length: {self.freqs_cis.shape[0]}\n------------------------------------")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear): 
            std_dev = 0.02 / math.sqrt(2 * self.config.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        _batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis_buffer = self.freqs_cis
        if seq_len > freqs_cis_buffer.shape[0]:
            print(f"Warn: Recomputing RoPE for seqlen {seq_len} > buffer {freqs_cis_buffer.shape[0]}")
            current_freqs_cis = precompute_freqs_cis(self.config.model_dim // self.config.num_heads, seq_len, theta=self.config.rope_theta).to(h.device)
        else: 
            current_freqs_cis = freqs_cis_buffer[:seq_len]

        for layer in self.layers: 
            h = layer(h, current_freqs_cis, attention_mask)

        h = self.norm(h)
        output_logits = self.output(h)
        return output_logits.float()


# --- Diffusion Helper Functions ---
def forward_masking(x0: torch.Tensor, t: torch.Tensor, mask_token_id: int, pad_token_id: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Applies random masking based on probability t. """
    device = x0.device
    bsz, seqlen = x0.shape
    is_pad = (x0 == pad_token_id)
    t = t.view(bsz, 1).to(device)
    rand_probs = torch.rand(bsz, seqlen, device=device)
    should_mask = (rand_probs < t) & (~is_pad)
    xt = torch.where(should_mask, mask_token_id, x0)
    
    return xt, should_mask

def calculate_masked_diffusion_loss(model: nn.Module, x0: torch.Tensor, attention_mask: torch.Tensor, config: CONFIG) -> torch.Tensor:
    """ Calculates the LLaDA diffusion loss (Eq 3). """
    batch_size, seq_len = x0.shape
    device = x0.device
    
    t = torch.rand(batch_size, device=device) * (1.0 - config.min_mask_prob) + config.min_mask_prob
    xt, was_masked = forward_masking(x0, t, config.mask_token_id, config.pad_token_id, config.vocab_size)
    logits = model(xt, attention_mask=attention_mask) 
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(logits.view(-1, config.vocab_size), x0.view(-1)).view(batch_size, seq_len)
    relevant_mask = was_masked & attention_mask.bool()
    masked_token_losses = token_losses * relevant_mask.float()
    sequence_loss_sum = masked_token_losses.sum(dim=1)
    t_clamped = t.clamp(min=config.loss_t_clamp_min)
    scaled_sequence_loss = sequence_loss_sum / t_clamped.squeeze(-1)
    batch_loss = scaled_sequence_loss.mean()
    if not torch.isfinite(batch_loss): 
        print(f"\nWARNING: Non-finite loss ({batch_loss.item()})")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return batch_loss


# --- Diffusion Inference Function ---
@torch.no_grad()
def run_conditional_diffusion_inference(
    model: nn.Module, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
    response_len: int, config: CONFIG, num_steps: int, verbose: bool = False
) -> str:
    """ Performs iterative conditional diffusion sampling (Random Remasking - Algo 4). """
    model.eval()
    device = config.device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    prompt_len = prompt_ids.shape[1]
    #total_len = prompt_len + response_len

    amp_context = torch.cuda.amp.autocast(enabled=config.use_amp, dtype=config.amp_dtype) if config.device == 'cuda' else nullcontext()

    # --- Step 1: Initialization ---
    response_part_r1 = torch.full((1, response_len), config.mask_token_id, dtype=torch.long, device=device)
    rt = torch.cat([prompt_ids, response_part_r1], dim=1)
    response_attn_mask = torch.ones_like(response_part_r1)
    combined_mask = torch.cat([prompt_mask, response_attn_mask], dim=1)

    if verbose: print(f"   [Step 0] Initial Response (IDs): {rt[0, prompt_len:].tolist()}")

    # --- Step 2: Time Schedule ---
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_val = timesteps[i]
        s_val = timesteps[i+1]

        with amp_context: logits = model(rt, attention_mask=combined_mask)
        pred_x0_all = torch.argmax(logits, dim=-1)

        remask_prob = s_val / t_val if t_val > 0 else 0.0; rand_uniform = torch.rand_like(rt, dtype=torch.float32)

        response_rt = rt[:, prompt_len:]
        is_masked_in_response = (response_rt == config.mask_token_id)
        pred_x0_response = pred_x0_all[:, prompt_len:]
        
        keep_mask_decision = (rand_uniform[:, prompt_len:] < remask_prob) & is_masked_in_response
        rs_response = torch.where(is_masked_in_response, torch.where(keep_mask_decision, config.mask_token_id, pred_x0_response), response_rt)
        rs = torch.cat([prompt_ids, rs_response], dim=1)
        rt = rs

        if verbose: #and (i % max(1, num_steps // 10) == 0 or i == num_steps - 1) 
            num_masks = (rt[0, prompt_len:] == config.mask_token_id).sum().item()
            intermediate_bytes = bytes([b for b in rt[0, prompt_len:].tolist() if 0 <= b <= 255])
            intermediate_text = intermediate_bytes.decode('utf-8', errors='replace')
            print(f"   [Step {i+1:>{len(str(num_steps))}}/{num_steps}] Masks: {num_masks:<4} | Resp: {repr(intermediate_text[:80])}...")

    # --- Step 4: Final Decode ---
    final_ids = rt
    generated_response_ids = final_ids[0, prompt_len:].tolist()
    valid_bytes = [byte_id for byte_id in generated_response_ids if 0 <= byte_id <= 255]
    byte_sequence = bytes(valid_bytes)
    
    try: 
        generated_text = byte_sequence.decode('utf-8', errors='replace')
    except Exception as e: 
        print(f"Warn: Decode Error: {e}")
        generated_text = f"DECODE_ERR:{repr(byte_sequence)}"
        
    model.train()
    return generated_text


def main():
    """ Main function: setup, training epochs, validation, saving. """
    print(f"--- Starting Byte-Level Model Training with AMP ---")
    print(f"Device: {CONFIG.device}, AMP Enabled: {CONFIG.use_amp}, AMP dtype: {CONFIG.amp_dtype}")
    overall_start_time = time.time() - 1744475000

    model = MaskPredictorLLaDA(CONFIG).to(CONFIG.device)
    #optimizer = SophiaG(model.parameters(), lr=CONFIG.learning_rate, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)
    scaler = GradScaler(enabled=CONFIG.use_amp)

    start_epoch = 0 
    global_step = 0

    # --- Prepare Data ---
    train_dataset = ByteLevelTextDataset(CONFIG.train_data_file, CONFIG.max_length, file_desc="Training")
    if not train_dataset.examples: 
        print("No training data. Exiting.") 
        return
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=CONFIG.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=CONFIG.num_workers, pin_memory=(CONFIG.device=='cuda'), persistent_workers=(CONFIG.num_workers > 0)
    )
    validation_sequences = load_validation_sequences(
        CONFIG.validation_data_file, CONFIG.max_length, CONFIG.num_validation_samples
    )

    # --- Training Loop ---
    print(f"Starting/Resuming training from Epoch {start_epoch+1}, Global Step {global_step}...")
    total_batches_in_dataset = len(train_dataloader)
    if total_batches_in_dataset == 0: 
        print("Training dataloader empty. Exiting.")
        return
    
    print(f"Total batches/epoch: {total_batches_in_dataset}")

    model.train()
    timestep = 0
    for epoch in range(start_epoch, CONFIG.num_epochs):
        epoch_start_time = time.time() - 1744475000
        epoch_loss_total = 0.0
        optimizer.zero_grad(set_to_none=True) # Zero grads at epoch start

        initial_pbar_step = global_step % total_batches_in_dataset
        pbar = tqdm(
            enumerate(train_dataloader), total=total_batches_in_dataset,
            desc=f"Epoch {epoch+1}/{CONFIG.num_epochs}", dynamic_ncols=True, initial=initial_pbar_step
        )
        if initial_pbar_step > 0: pbar.update(0)

        for batch_idx_in_epoch, batch in pbar:
            # --- Skip processed batches if resuming ---
            current_global_step_for_batch = epoch * total_batches_in_dataset + batch_idx_in_epoch
            if current_global_step_for_batch < global_step: continue

            input_ids = batch['input_ids'].to(CONFIG.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(CONFIG.device, non_blocking=True)
            if input_ids.numel() == 0:
                print(f"Warning: Skipping empty batch at E{epoch+1}, B{batch_idx_in_epoch}, G{global_step}")
                global_step += 1 
                continue

            # --- Determine Autocast Context ---
            amp_context = autocast(enabled=CONFIG.use_amp, dtype=CONFIG.amp_dtype) if CONFIG.device == 'cuda' else nullcontext()

            # --- Forward, Loss, Backward with AMP ---
            with amp_context:
                loss = calculate_masked_diffusion_loss(model, input_ids, attention_mask, CONFIG)

            if not torch.isfinite(loss):
                print(f"\nWarn: Non-finite loss E{epoch+1}, B{batch_idx_in_epoch}, G{global_step}. Skip update.")
                if (global_step + 1) % CONFIG.gradient_accumulation_steps == 0: 
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                continue

            scaled_loss = scaler.scale(loss / CONFIG.gradient_accumulation_steps)
            scaled_loss.backward()
            epoch_loss_total += loss.item() # Track original loss

            timestep += 1
            writer.add_scalar('Loss', loss, timestep)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], timestep)

            # --- Optimizer Step ---
            if (global_step + 1) % CONFIG.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Scale": f"{scaler.get_scale():.1f}", "Step": global_step + 1})

            global_step += 1

            # --- Periodic Logging ---
            if global_step > 0 and global_step % CONFIG.logging_steps == 0:
                print(f"\nGlobal Step: {global_step}, Last Batch Loss: {loss.item():.4f}") # Simple log

            # --- Periodic Validation & Saving ---
            if global_step > 0 and global_step % CONFIG.predict_steps == 0:
                if validation_sequences:
                    print("\n" + "="*20 + f" Validation @ GStep {global_step} " + "="*20)
                    for i, val_seq_data in enumerate(validation_sequences):
                        full_ids = val_seq_data["input_ids"].to(CONFIG.device)
                        full_mask = val_seq_data["attention_mask"].to(CONFIG.device)
                        seq_len = full_ids.shape[1]

                        min_prompt = min(CONFIG.val_prompt_min_len, seq_len - 1 if seq_len > 1 else 1)
                        max_prompt = min(CONFIG.val_prompt_max_len, seq_len - 1 if seq_len > 1 else 1)
                        prompt_len = random.randint(min_prompt, max_prompt) if min_prompt < max_prompt else min_prompt

                        max_possible_resp_len = CONFIG.rope_max_length_buffer - prompt_len # Limit by RoPE buffer
                        max_resp = min(CONFIG.val_response_max_len, max(1, max_possible_resp_len))
                        min_resp = min(CONFIG.val_response_min_len, max(1, max_resp -1))
                        response_len_to_generate = random.randint(min_resp, max_resp) if min_resp < max_resp else min_resp
                        if prompt_len <= 0 or response_len_to_generate <= 0: 
                            print(f"Warn: Invalid random split val {i+1}. Skip.")
                            continue

                        prompt_ids_val = full_ids[:, :prompt_len]
                        prompt_mask_val = full_mask[:, :prompt_len]
                        target_response_ids = full_ids[:, prompt_len:]
                        target_response_mask = full_mask[:, prompt_len:]

                        prompt_bytes = bytes([b for b in prompt_ids_val[0].tolist() if 0 <= b <= 255])
                        target_bytes = bytes([b for b in target_response_ids[0, target_response_mask[0] == 1].tolist() if 0 <= b <= 255])
                        prompt_text_repr = repr(prompt_bytes.decode('utf-8', errors='replace'))
                        target_text_repr = repr(target_bytes.decode('utf-8', errors='replace'))
                        
                        is_verbose = CONFIG.verbose_inference

                        print(f"\n--- Val Sample {i+1} (Prompt Len: {prompt_len}, Gen Len: {response_len_to_generate}) ---")
                        print(f"Prompt: {prompt_text_repr}")
                        print(f"Target: {target_text_repr} (Original suffix for reference)")

                        generated_text_raw = run_conditional_diffusion_inference(model, prompt_ids_val, prompt_mask_val, response_len_to_generate, CONFIG, CONFIG.inference_steps, verbose=is_verbose)
                        generated_text_repr = repr(generated_text_raw) # Show raw output

                        print(f"Gen:    {generated_text_repr}")
                        if is_verbose: print("-" * 15 + " End Verbose " + "-" * 15)
                    print("="*60 + "\n")
                    model.train() 
                else: 
                    print(f"\nStep {global_step}: No validation sequences loaded.")


        epoch_duration = time.time() - 1744475000 - epoch_start_time
        avg_epoch_loss = epoch_loss_total / total_batches_in_dataset if total_batches_in_dataset > 0 else 0.0
        print(f"\n--- Epoch {epoch+1} Finished ---")
        print(f"Time Taken: {epoch_duration:.2f} seconds")
        print(f"Average Training Loss for Epoch: {avg_epoch_loss:.4f}")
        print(f"Global Step Reached: {global_step}")
        print("---------------------------\n")


    total_training_time = time.time() - 1744475000 - overall_start_time
    print("--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")

if __name__ == "__main__":
    writer = SummaryWriter()
    writer = SummaryWriter(CONFIG.name) 
    writer = SummaryWriter(comment=f"{CONFIG.name} Larger Slower Lower Batch Size test7.py {time.time() - 1744475000}") 
    
    train_file = Path(CONFIG.train_data_file)
    if not train_file.exists(): print(f"{train_file} not found. Dummy file will be created by dataset loader.")
    validation_file = Path(CONFIG.validation_data_file)
    if not validation_file.exists():
        print(f"Creating dummy validation file: {validation_file}")
        dummy_val_text = ("Validation sentence one.\nHas newline.\n" * 5)
        try: validation_file.write_bytes(dummy_val_text.encode('utf-8', errors='replace'))
        except Exception as e: print(f"Error creating dummy validation file: {e}")

    main()
