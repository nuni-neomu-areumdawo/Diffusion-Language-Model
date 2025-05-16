"""
Implementation of a LLaDA-inspired Masked Diffusion Model for Text
using PURE BYTE-LEVEL TOKENIZATION and Mixed Precision Training.
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
from contextlib import nullcontext
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter #tensorboard --logdir=runs
import mmap
import os

#For more predictable results.
torch.manual_seed(0)
random.seed(0)

#Change stuff here
class CONFIG:
    """ Stores configuration parameters for the model, training, and data """

    #MLA Test
    head_dim = 32
    v_head_dim = head_dim
    nope_head_dim = 32
    rope_head_dim = 64

    kv_lora_rank = 64
    q_lora_rank = 4 * kv_lora_rank

    #Special Byte-Level Token IDs
    PAD_ID: int = 256
    MASK_ID: int = 257
    UNK_ID: int = 258
    NUM_SPECIAL_TOKENS: int = 3

    #Data Files & Dirs
    train_data_file: str = "./concat.txt"
    validation_data_file: str = "./validation.txt"
    checkpoint_dir: Path = Path("./llada_byte_checkpoints_amp_v4")
    final_model_dir: Path = Path("./llada_byte_model_final_amp_v4")
    name = "Diffusion-LM"

    # --- Model Architecture ---
    vocab_size: int = 256 + NUM_SPECIAL_TOKENS
    model_dim: int = 768 #Higher = better memorization (good range is 128-2048~)
    num_layers: int = 16 #Higher = better logic but harder to train (good range is 4-32~)
    num_heads: int = 16
    kv_latent_dim: int = model_dim // 16 # Example: 1024 // 4 = 256
    dropout = 0.25

    assert kv_latent_dim > 0 and kv_latent_dim <= model_dim, "kv_latent_dim must be positive and <= model_dim"

    ffn_dim_multiplier: Optional[float] = None
    multiple_of: int = 256 #Ensure it aligns with model_dim
    norm_eps: float = 1e-6 #1e-5
    use_bias: bool = False # Bias in Attention/FFN layers

    #RoPE Specific
    rope_theta: float = 10000.0
    rope_max_length_buffer: int = 2048 # Max length for RoPE precomputation buffer - should be >= max(train_max_len, max_infer_len)

    #Training
    max_length: int = 768 # Max length for *training* chunks (in bytes), this increases model requirements.
    batch_size: int = 8 #Lower if your computer suffers
    learning_rate: float = 2e-4
    num_epochs: int = 500 # Train for this many epochs
    logging_steps: int = 250 # Log loss every N steps
    predict_steps: int = 2500 # Run validation inference every N steps
    gradient_accumulation_steps: int = 2
    gradient_clip_val: float = 1.0 # Max norm for gradient clipping
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8 # Set num_workers=0 to debug DataLoader issues causing None items in collate_fn

    #Mixed Precision
    use_amp: bool = torch.cuda.is_available()

    # Use bfloat16 if available, otherwise float16 for AMP
    amp_dtype: torch.dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    #Diffusion
    mask_token_id: int = MASK_ID
    pad_token_id: int = PAD_ID
    min_mask_prob: float = 0.05 # Min 't' during training
    loss_t_clamp_min: float = 1e-6 # Min 't' for loss division

    #Validation
    inference_steps: int = 128 # Number of diffusion steps for validation samples
    num_validation_samples: int = 5 # Number of validation examples to genearte

    #Random length ranges for validation inference

    val_prompt_min_len: int = 256
    val_prompt_max_len: int = 768
    val_response_min_len: int = 128
    val_response_max_len: int = 256

    verbose_inference: bool = True # Show step-by-step denoising?
    repetition_penalty_strength: float = 1.05  # Values > 1.0 penalize. 1.0 means no penalty. Tune this.
    repetition_penalty_window: int = 8
    
    consecutive_repetition_threshold: int = 3 # e.g., "    " (4 spaces)
    repetitive_remask_boost: float = 1.0032 # e.g., 10% boost
    min_len_for_targeted_remask: int = 8


# --- Tokenizer Info (Byte Handling) ---
print("--- Using Byte-Level Tokenization ---")
print(f"Byte range: 0-255")
print(f"Special Tokens: PAD={CONFIG.PAD_ID}, MASK={CONFIG.MASK_ID}, UNK={CONFIG.UNK_ID}")
print(f"Effective Vocab Size: {CONFIG.vocab_size}")
if CONFIG.MASK_ID is None or CONFIG.PAD_ID is None:
    raise ValueError("Essential special tokens (MASK_ID, PAD_ID) are not defined.")


class ByteLevelTextDataset(Dataset):
    """
    Loads text data from a file as raw bytes using memory mapping for streaming,
    converts bytes to their integer representations (0-255) on-the-fly,
    and provides fixed-length sequences.
    Handles multiprocessing by initializing mmap in each worker.
    """
    def __init__(
        self, file_path: Union[str, Path], max_length: int, file_desc: str = "Training"
    ):
        self.max_length = max_length
        self.file_path = Path(file_path)
        self.file_desc = file_desc
        print(f"Initializing {self.file_desc} dataset (streaming) from '{self.file_path}' (main process or worker init)...")

        self.file = None
        self.mmap_obj = None
        self._length: int = 0
        self.file_size: int = 0

        try:
            if not self.file_path.exists() or self.file_path.stat().st_size == 0:
                if self.file_desc == "Training":
                    print(f"Warning: {self.file_desc} file '{self.file_path}' empty or not found in __init__.")
                    self._create_dummy_byte_file()
                    if not self.file_path.exists() or self.file_path.stat().st_size == 0:
                        print(f"Error: Dummy file creation failed or resulted in an empty file for {self.file_path}.")
                        return
                else:
                    print(f"Warning: {self.file_desc} file '{self.file_path}' empty or not found. No examples will be loaded.")
                    return

            self.file_size = self.file_path.stat().st_size
            if self.file_size == 0:
                print(f"Warning: {self.file_desc} file '{self.file_path}' is empty. No sequences will be created.")
                return
            if self.file_size < self.max_length:
                print(f"Warning: {self.file_desc} file '{self.file_path}' (size: {self.file_size}) "
                    f"is smaller than max_length ({self.max_length}). No sequences will be created.")
                return

            self._length = self.file_size // self.max_length

            if self._length == 0:
                print(f"WARNING: No full sequences can be created for {self.file_desc}.\nContent length {self.file_size} < max_length {self.max_length}?")
            else:
                pass
        except FileNotFoundError:
            print(f"FileNotFoundError during __init__ for {self.file_path} (should have been caught or dummy created).")
        except Exception as e:
            print(f"Error during initial setup of dataset '{self.file_path}': {e}")
            self._cleanup_resources()

    def _create_dummy_byte_file(self):
        print(f"Creating dummy byte data file: {self.file_path}")
        dummy_text = ("Dummy sentence for training.\nIncludes newlines.\n" * 20)
        dummy_bytes_segment = dummy_text.encode('utf-8', errors='replace')
        if not dummy_bytes_segment:
            print("Error: Dummy byte segment is empty.")
            return
        num_chunks_to_create = 5
        required_bytes = self.max_length * num_chunks_to_create
        repeats = (required_bytes // len(dummy_bytes_segment)) + 1
        final_dummy_bytes = dummy_bytes_segment * repeats
        try:
            self.file_path.write_bytes(final_dummy_bytes)
            print(f"Dummy file created: '{self.file_path}' with approx {len(final_dummy_bytes)} bytes.")
        except IOError as e:
            print(f"Error creating dummy byte file '{self.file_path}': {e}")

    def _ensure_resources_open(self):
        """Opens file and mmap object if they haven't been opened yet (e.g., in a new worker)."""
        if self.mmap_obj is None:
            try:
                if not self.file_path.exists() or self.file_path.stat().st_size == 0:
                    print(f"Error in worker: {self.file_desc} file '{self.file_path}' not found or empty when trying to mmap.")
                    self._length = 0
                    return False

                current_file_size = self.file_path.stat().st_size
                if current_file_size < self.max_length:
                    print(f"Error in worker: {self.file_desc} file '{self.file_path}' (size: {current_file_size}) "
                        f"too small for max_length ({self.max_length}).")
                    self._length = 0
                    return False

                self.file = open(self.file_path, "rb")
                self.mmap_obj = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
                
                self.file_size = current_file_size
                self._length = self.file_size // self.max_length

                if self._length > 0:
                    print(f"Process {os.getpid()}: Successfully mmapped {self.file_path}, length: {self._length}")
                else:
                    print(f"Process {os.getpid()}: Mmapped {self.file_path}, but no full sequences. File size: {self.file_size}, max_length: {self.max_length}")
                    self._cleanup_resources() 
                    return False
                return True

            except Exception as e:
                print(f"Error opening/mmapping file '{self.file_path}' in process {os.getpid()}: {e}")
                self._cleanup_resources()
                self._length = 0
                return False
        return True

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        if not self._ensure_resources_open() or self.mmap_obj is None:
            return None

        if not (0 <= idx < self._length):
            return None

        offset = idx * self.max_length
        try:
            byte_chunk = self.mmap_obj[offset : offset + self.max_length]
            byte_ids_list = list(byte_chunk)
            return {"input_ids": torch.tensor(byte_ids_list, dtype=torch.long)}
        except ValueError as ve: 
            print(f"ValueError reading chunk at index {idx} from {self.file_desc} mmap (possibly closed or offset issue): {ve}, pid {os.getpid()}")
            self._cleanup_resources() 
            return None
        except Exception as e:
            print(f"Error reading chunk at index {idx} from {self.file_desc} mmap: {e}, pid {os.getpid()}")
            return None

    def _cleanup_resources(self):
        # print(f"Process {os.getpid()}: Cleaning up resources for {self.file_path}")
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception as e:
                print(f"Error closing mmap_obj for {self.file_path} in {os.getpid()}: {e}")
            self.mmap_obj = None
        if self.file:
            try:
                self.file.close()
            except Exception as e:
                print(f"Error closing file for {self.file_path} in {os.getpid()}: {e}")
            self.file = None

    def close(self):
        self._cleanup_resources()

    def __del__(self):
        self._cleanup_resources()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state['mmap_obj'] = None

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

def collate_fn(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of byte sequences, handling potential None items.
    Pads sequences and creates an attention mask.
    """
    valid_items = [item for item in batch if item is not None and 'input_ids' in item]

    if not valid_items:
        #print("Warning: Collate function received empty or all-None/invalid batch.")
        return {"input_ids": torch.empty((0, CONFIG.max_length if hasattr(CONFIG, 'max_length') else 0), dtype=torch.long), # Ensure shape if possible
                "attention_mask": torch.empty((0, CONFIG.max_length if hasattr(CONFIG, 'max_length') else 0), dtype=torch.long)}

    input_ids = [item['input_ids'] for item in valid_items]

    #if any(t.size(0) != CONFIG.max_length for t in input_ids):
    #print(f"Warning: Inconsistent tensor lengths in collate_fn: {[t.size(0) for t in input_ids]}")


    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=CONFIG.pad_token_id
    )
    attention_mask = (padded_input_ids != CONFIG.pad_token_id).long()
    return {"input_ids": padded_input_ids, "attention_mask": attention_mask}


def load_validation_sequences(
    file_path: Union[str, Path], max_length: int, num_samples: int
) -> List[Dict[str, torch.Tensor]]:
    print(f"Loading Validation Byte Sequences ({num_samples} samples)")
    validation_sequences = []
    val_dataset = None
    try:
        val_dataset = ByteLevelTextDataset(file_path, max_length, file_desc="Validation")
        if len(val_dataset) == 0: # Check if dataset could be initialized properly
            print("No validation sequences to load (num_samples or dataset length is 0).")
            return []

        num_to_load = min(num_samples, len(val_dataset))
        if num_to_load == 0:
            print("No validation sequences to load (num_samples or dataset length is 0).")
            return []

        for i in range(num_to_load):
            sample = val_dataset[i]
            if sample is None:
                print(f"Warning: Validation sample at index {i} is None.")
                continue
            ids = sample['input_ids']
            mask = (ids != CONFIG.pad_token_id).long().unsqueeze(0)
            ids = ids.unsqueeze(0)
            validation_sequences.append({"input_ids": ids, "attention_mask": mask})
        print(f"Successfully loaded {len(validation_sequences)} validation byte sequences.")
    except Exception as e:
        print(f"Error loading validation sequences: {e}")
        validation_sequences = []
    finally:
        if val_dataset:
            val_dataset.close() # Explicitly close the validation dataset's resources
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
        assert config.nope_head_dim is not None, f"nope_head_dim is not defined {config.nope_head_dim=}"


        self.config = config

        self.dim = config.model_dim
        self.num_heads = config.num_heads
        self.v_head_dim = config.v_head_dim

        self.nope_head_dim = config.nope_head_dim 
        self.rope_head_dim = config.rope_head_dim

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

        self.dropout_p = config.dropout 

        self.value_out_dim = self.num_heads * self.v_head_dim

        self.nope_q_dim_total = self.num_heads * self.nope_head_dim
        self.nope_k_dim_total = self.num_heads * self.nope_head_dim 

        self.rope_q_dim_total = self.num_heads * self.rope_head_dim

        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=self.config.use_bias)
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_q_dim_total, bias=self.config.use_bias)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_q_dim_total, bias=self.config.use_bias)
        self.q_norm = RMSNorm(self.q_lora_rank) 

        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=self.config.use_bias)
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_k_dim_total, bias=self.config.use_bias)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_out_dim, bias=self.config.use_bias)
        self.kv_norm = RMSNorm(self.kv_lora_rank) # Norm is applied to the compressed KV

        # K_rope is projected from full dimension 'x' directly to rope_head_dim 
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim, bias=self.config.use_bias)

        self.proj = nn.Linear(self.value_out_dim, self.dim, bias=self.config.use_bias) 
        if self.dropout_p > 0:
            self.res_dropout = nn.Dropout(p=self.dropout_p)


    def forward(self, x: Tensor, freqs_cis: Tensor, padding_mask: Optional[torch.Tensor]): 
        batch_size, seq_len, _ = x.shape

        # Query processing
        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope_flat = self.decompress_q_nope(norm_q) 
        query_rope_flat = self.decompress_q_rope(norm_q)

        # Key and Value processing
        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope_flat = self.decompress_k_nope(norm_kv)
        value_flat = self.decompress_v_linear(norm_kv)

        # Key RoPE part processing (projected from original x)
        key_rope_flat_single_head = self.k_rope_linear(x)

        # Reshape for multi-head attention
        query_nope = query_nope_flat.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope_flat.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)

        # Key NoPE: (bs, num_heads, seq_len, nope_head_dim)
        key_nope = key_nope_flat.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)

        key_rope_single_head_view = key_rope_flat_single_head.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)

        key_rope_to_embed = key_rope_single_head_view / self.num_heads 
        
        value = value_flat.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1,2)

        current_freqs_cis = freqs_cis[:seq_len] 

        q_rope_embedded, k_rope_embedded = apply_rotary_emb(query_rope, key_rope_to_embed, current_freqs_cis)

        if k_rope_embedded.shape[1] == 1 and self.num_heads > 1:
            k_rope_embedded = k_rope_embedded.expand(-1, self.num_heads, -1, -1)

        combined_head_dim = self.nope_head_dim + self.rope_head_dim
        q_recombined = torch.cat([query_nope, q_rope_embedded], dim=-1)
        k_recombined = torch.cat([key_nope, k_rope_embedded], dim=-1)

        attn_mask_for_pytorch = None
        if padding_mask is not None:
            attn_mask_for_pytorch = (1.0 - padding_mask[:, None, None, :].float()) * torch.finfo(q_recombined.dtype).min

        output = F.scaled_dot_product_attention(
            q_recombined, k_recombined, value,
            attn_mask=attn_mask_for_pytorch, 
            is_causal=False, # LLaDA is bidirectional
            dropout_p=self.dropout_p if self.training else 0.0
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.value_out_dim)

        output = self.proj(output)
        if self.dropout_p > 0:
            output = self.res_dropout(output)
        return output

class FeedForward(nn.Module):
    #Modified SwiGLU Feed-Forward Network.
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, multiple_of: int = 256, use_bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=use_bias)

    def forward(self, x):
        return self.w2(F.mish(self.w1(x), inplace=True) * self.w3(x)) #Mish has better performance change to SiLU for loyalty

class TransformerBlock(nn.Module):
    #Transformer block: Attention and FFN with pre-normalization.
    def __init__(self, config: CONFIG):
        super().__init__()

        self.attention = MultiHeadLatentAttention(config)

        self.feed_forward = FeedForward(dim=config.model_dim, multiple_of=config.multiple_of, use_bias=config.use_bias)
        self.attention_norm = RMSNorm(config.model_dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.model_dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:

        h = x + self.attention(self.attention_norm(x), freqs_cis, padding_mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class MaskPredictorLLaDA(nn.Module):
    def __init__(self, config: CONFIG):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.model_dim, eps=config.norm_eps)
        self.output = nn.Linear(config.model_dim, config.vocab_size, bias=config.use_bias)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=config.rope_head_dim, 
                end=config.rope_max_length_buffer,
                theta=config.rope_theta
            ),
            persistent=False
        )

        self.apply(self._init_weights)
        print(f"--- Byte-Level Model Initialized (Using Full MLA) ---")
        print(f"Vocab Size: {config.vocab_size}, Layers: {config.num_layers}, Dim: {config.model_dim}, Heads: {config.num_heads}")
        print(f"MLA Config: RoPE Dim: {config.rope_head_dim}, NoPE Dim: {config.nope_head_dim}, V Dim: {config.v_head_dim}")
        print(f"Params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print(f"RoPE Buffer Length: {self.freqs_cis.shape[0]} for dim {config.rope_head_dim}")
        print("------------------------------------")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std_dev = 0.02 / math.sqrt(2 * self.config.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std_dev)
            #if module.bias is not None:
            #torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        _batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis_buffer = self.freqs_cis
        if seq_len > freqs_cis_buffer.shape[0]:
            print(f"Warning: Sequence length {seq_len} exceeds RoPE precomputed buffer {freqs_cis_buffer.shape[0]}. Recomputing RoPE frequencies for dim {self.config.rope_head_dim}.")
            current_freqs_cis = precompute_freqs_cis(
                dim=self.config.rope_head_dim,
                end=seq_len,
                theta=self.config.rope_theta
            ).to(h.device)
        else:
            current_freqs_cis = freqs_cis_buffer[:seq_len]

        for layer in self.layers:
            h = layer(h, current_freqs_cis, attention_mask)

        h = self.norm(h)
        output_logits = self.output(h)
        return output_logits.float()


# --- Diffusion Helper Functions ---
def forward_masking(x0: torch.Tensor, t: torch.Tensor, mask_token_id: int, pad_token_id: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #Applies random masking based on probability t.
    device = x0.device
    bsz, seqlen = x0.shape
    is_pad = (x0 == pad_token_id)
    t_expanded = t.view(bsz, 1).to(device)
    rand_probs = torch.rand(bsz, seqlen, device=device)
    should_mask = (rand_probs < t_expanded) & (~is_pad)
    xt = torch.where(should_mask, mask_token_id, x0)

    return xt, should_mask

def calculate_masked_diffusion_loss(model: nn.Module, x0: torch.Tensor, attention_mask: torch.Tensor, config: CONFIG) -> torch.Tensor:
    batch_size, seq_len = x0.shape
    device = x0.device

    t = torch.rand(batch_size, device=device) * (1.0 - config.min_mask_prob) + config.min_mask_prob
    xt, was_masked = forward_masking(x0, t, config.mask_token_id, config.pad_token_id, config.vocab_size)

    logits = model(xt, attention_mask=attention_mask) 

    loss_fct = nn.CrossEntropyLoss(reduction='none') 

    token_losses = loss_fct(logits.view(-1, config.vocab_size), x0.view(-1)).view(batch_size, seq_len)

    relevant_loss_mask = was_masked & attention_mask.bool() 

    masked_token_losses = token_losses * relevant_loss_mask.float()
    sequence_loss_sum = masked_token_losses.sum(dim=1)

    t_clamped = t.clamp(min=config.loss_t_clamp_min)
    scaled_sequence_loss = sequence_loss_sum / t_clamped 

    batch_loss = scaled_sequence_loss.mean()

    if not torch.isfinite(batch_loss):
        print(f"\nWARNING: Non-finite loss ({batch_loss.item()}). Details: "
            # f"Seq Loss Sum: {sequence_loss_sum}, t_clamped: {t_clamped}, "
            # f"Token losses min/max: {token_losses.min()}/{token_losses.max()}"
            )
        return torch.tensor(0.0, device=device, requires_grad=True) 

    return batch_loss

#Diffusion Inference Function
@torch.no_grad()
def run_conditional_diffusion_inference(
    model: nn.Module, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
    response_len: int, config: CONFIG, num_steps: int, verbose: bool = False
) -> str:
    model.eval()
    device = config.device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    prompt_len = prompt_ids.shape[1]

    amp_context = torch.cuda.amp.autocast(enabled=config.use_amp, dtype=config.amp_dtype) if config.device == 'cuda' else nullcontext()

    response_part_r1 = torch.full((1, response_len), config.mask_token_id, dtype=torch.long, device=device)
    rt = torch.cat([prompt_ids, response_part_r1], dim=1) # rt shape: (1, total_len)
    response_attn_mask = torch.ones_like(response_part_r1)
    combined_mask = torch.cat([prompt_mask, response_attn_mask], dim=1)

    if verbose:
        print(f"   [Step 0] Initial Response (IDs): {rt[0, prompt_len:].tolist()}")

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_val = timesteps[i].item()
        s_val = timesteps[i+1].item()

        with amp_context:
            logits = model(rt, attention_mask=combined_mask)

        #Apply General Repetition Penalty
        if config.repetition_penalty_strength > 1.0 and logits.shape[0] == 1:
            modified_logits = logits.clone()
            for j in range(response_len): # Iterate over each position in the response part of the logits
                current_pred_position_in_total = prompt_len + j
                start_idx_penalty_window = 0
                if config.repetition_penalty_window > 0:
                    start_idx_penalty_window = max(0, current_pred_position_in_total - config.repetition_penalty_window)
                
                ids_to_penalize = set()
                if current_pred_position_in_total > start_idx_penalty_window:
                    context_slice = rt[0, start_idx_penalty_window:current_pred_position_in_total]
                    for token_id_tensor in context_slice:
                        token_id = token_id_tensor.item()
                        if 0 <= token_id <= 255: # Only penalize actual byte tokens
                            ids_to_penalize.add(token_id)
                
                if ids_to_penalize:
                    for token_val_to_penalize in ids_to_penalize:
                        logit_val = modified_logits[0, current_pred_position_in_total, token_val_to_penalize]
                        if logit_val > 0:
                            modified_logits[0, current_pred_position_in_total, token_val_to_penalize] /= config.repetition_penalty_strength
                        else:
                            modified_logits[0, current_pred_position_in_total, token_val_to_penalize] *= config.repetition_penalty_strength
            logits = modified_logits
        
        pred_x0_all = torch.argmax(logits, dim=-1) # Predicted x0 based on (potentially penalized) logits

        base_remask_prob_scalar = s_val / t_val if t_val > 0 else 0.0
        current_response_part_rt = rt[0, prompt_len:]
        
        remask_probabilities = torch.full_like(current_response_part_rt, base_remask_prob_scalar, dtype=torch.float32)
        
        if response_len >= config.min_len_for_targeted_remask and config.consecutive_repetition_threshold > 1:
            predicted_response_tokens = pred_x0_all[0, prompt_len:].tolist()
            count = 0
            last_token = -1
            
            for k in range(response_len):
                current_token = predicted_response_tokens[k]
                if 0 <= current_token <= 255:
                    if current_token == last_token:
                        count += 1
                    else:
                        if count >= config.consecutive_repetition_threshold:
                            for l_idx in range(k - count, k):
                                if current_response_part_rt[l_idx].item() == config.mask_token_id:
                                    remask_probabilities[l_idx] = min(1.0, base_remask_prob_scalar * config.repetitive_remask_boost)
                        count = 1
                        last_token = current_token
                else:
                    if count >= config.consecutive_repetition_threshold:
                        for l_idx in range(k - count, k):
                            if current_response_part_rt[l_idx].item() == config.mask_token_id:
                                remask_probabilities[l_idx] = min(1.0, base_remask_prob_scalar * config.repetitive_remask_boost)
                    count = 0
                    last_token = -1
            
            if count >= config.consecutive_repetition_threshold:
                for l_idx in range(response_len - count, response_len):
                    if current_response_part_rt[l_idx].item() == config.mask_token_id:
                            remask_probabilities[l_idx] = min(1.0, base_remask_prob_scalar * config.repetitive_remask_boost)

        rand_uniform_response = torch.rand((1, response_len), device=device, dtype=torch.float32)

        response_rt_current = rt[:, prompt_len:]
        pred_x0_response = pred_x0_all[:, prompt_len:]
        is_masked_in_response = (response_rt_current == config.mask_token_id)
        
        keep_mask_decision = (rand_uniform_response < remask_probabilities) & is_masked_in_response
                
        rs_response = torch.where(
            is_masked_in_response,
            torch.where(keep_mask_decision, config.mask_token_id, pred_x0_response),
            response_rt_current 
        )
        rt = torch.cat([prompt_ids, rs_response], dim=1)

        if verbose:
            num_masks = (rt[0, prompt_len:] == config.mask_token_id).sum().item()
            avg_remask_p_display = remask_probabilities.mean().item() if response_len > 0 else base_remask_prob_scalar
            intermediate_bytes = bytes([b for b in rt[0, prompt_len:].tolist() if 0 <= b <= 255])
            intermediate_text = intermediate_bytes.decode('utf-8', errors='replace')
            print(f"   [Step {i+1:>{len(str(num_steps))}}/{num_steps}] avg_remask_p={avg_remask_p_display:.3f} Masks: {num_masks:<4} | Gen: {repr(intermediate_text[:100])}...")

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
    print(f"Number of workers: {CONFIG.num_workers}")
    overall_start_time = time.time()

    model = MaskPredictorLLaDA(CONFIG).to(CONFIG.device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25000)
    scaler = GradScaler(enabled=CONFIG.use_amp)

    start_epoch = 0
    global_step = 0

    train_dataset = ByteLevelTextDataset(CONFIG.train_data_file, CONFIG.max_length, file_desc="Training")
    if len(train_dataset) == 0:
        print("No training data available (dataset length is 0 after init). Exiting.")
        return 
    
    use_persistent_workers = (CONFIG.num_workers > 0)
    train_dataloader = DataLoader(
        train_dataset, batch_size=CONFIG.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=CONFIG.num_workers,
        pin_memory=(CONFIG.device=='cuda'),
        persistent_workers=use_persistent_workers,
    )
    print(f"Train DataLoader: persistent_workers={use_persistent_workers}")


    validation_sequences = load_validation_sequences(
        CONFIG.validation_data_file, CONFIG.max_length, CONFIG.num_validation_samples
    )

    print(f"Starting/Resuming training from Epoch {start_epoch+1}, Global Step {global_step}...")
    if len(train_dataloader) == 0 :
        print("Training dataloader is empty. Exiting.")
        return
    total_batches_in_dataset = len(train_dataloader)
    print(f"Total batches/epoch: {total_batches_in_dataset}")

    model.train()
    timestep_tb = 0
    for epoch in range(start_epoch, CONFIG.num_epochs):
        epoch_start_time = time.time()
        epoch_loss_total = 0.0

        pbar = tqdm(
            train_dataloader,
            total=total_batches_in_dataset,
            desc=f"Epoch {epoch+1}/{CONFIG.num_epochs}",
            dynamic_ncols=True,
        )

        for batch_idx_in_epoch, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(CONFIG.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(CONFIG.device, non_blocking=True)

            if input_ids.numel() == 0 or attention_mask.numel() == 0:
                print(f"Warning: Skipping empty batch (numel is 0) at E{epoch+1}, B{batch_idx_in_epoch}, G{global_step}")
                continue


            amp_context = autocast(enabled=CONFIG.use_amp, dtype=CONFIG.amp_dtype) if CONFIG.device == 'cuda' else nullcontext()

            with amp_context:
                loss = calculate_masked_diffusion_loss(model, input_ids, attention_mask, CONFIG)

            if not torch.isfinite(loss):
                print(f"\nWarning: Non-finite loss {loss.item()} encountered at E{epoch+1}, B{batch_idx_in_epoch}, G{global_step}. Skipping update for this batch.")
                if (global_step + 1) % CONFIG.gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            epoch_loss_total += loss.item() * CONFIG.gradient_accumulation_steps

            scaled_loss = scaler.scale(loss / CONFIG.gradient_accumulation_steps)
            scaled_loss.backward()


            if (global_step + 1) % CONFIG.gradient_accumulation_steps == 0:
                writer.add_scalar('Loss/train_step', loss.item(), global_step) 
                writer.add_scalar('LearningRate/step', optimizer.param_groups[0]['lr'], global_step)

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) 
                scheduler.step()
                
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "AvgLoss": f"{epoch_loss_total / ((batch_idx_in_epoch+1)*CONFIG.gradient_accumulation_steps):.4f}",
                    "Scale": f"{scaler.get_scale():.1f}",
                    "Step": global_step + 1
                })
            global_step += 1
            timestep_tb += 1

            if global_step > 0 and (global_step % (CONFIG.logging_steps * CONFIG.gradient_accumulation_steps) == 0):
                avg_loss_so_far = epoch_loss_total / (batch_idx_in_epoch + 1) / CONFIG.gradient_accumulation_steps
                print(f"\nGlobal Step: {global_step}, Avg Loss Last {CONFIG.logging_steps} steps (approx): {avg_loss_so_far:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.2e}")


            if global_step > 0 and (global_step % (CONFIG.predict_steps * CONFIG.gradient_accumulation_steps) == 0):
                if validation_sequences:
                    print("\n" + "="*20 + f" Validation at Global Step {global_step} " + "="*20)
                    model.eval()
                    for i, val_seq_data in enumerate(validation_sequences):
                        full_ids = val_seq_data["input_ids"].to(CONFIG.device)
                        full_mask = val_seq_data["attention_mask"].to(CONFIG.device)
                        seq_len_val = full_ids.shape[1]

                        min_prompt = min(CONFIG.val_prompt_min_len, seq_len_val - 1 if seq_len_val > 1 else 1)
                        max_prompt = min(CONFIG.val_prompt_max_len, seq_len_val - 1 if seq_len_val > 1 else 1)
                        prompt_len = random.randint(min_prompt, max_prompt) if min_prompt < max_prompt else min_prompt
                        
                        max_possible_resp_len = CONFIG.rope_max_length_buffer - prompt_len
                        max_resp = min(CONFIG.val_response_max_len, max(1, max_possible_resp_len))
                        min_resp = min(CONFIG.val_response_min_len, max(1, max_resp -1 if max_resp > 1 else 1) )
                        response_len_to_generate = random.randint(min_resp, max_resp) if min_resp < max_resp else min_resp

                        if prompt_len <= 0 or response_len_to_generate <= 0:
                            print(f"Warn: Invalid random split for validation sample {i+1} (prompt {prompt_len}, resp {response_len_to_generate}). Skipping.")
                            continue

                        prompt_ids_val = full_ids[:, :prompt_len]
                        prompt_mask_val = full_mask[:, :prompt_len]
                        target_response_ids = full_ids[:, prompt_len:]
                        target_response_mask = full_mask[:, prompt_len:]

                        prompt_bytes = bytes([b for b in prompt_ids_val[0].tolist() if 0 <= b <= 255])
                        target_bytes = bytes([b for b in target_response_ids[0, target_response_mask[0] == 1].tolist() if 0 <= b <= 255]) # Use mask for target
                        prompt_text_repr = repr(prompt_bytes.decode('utf-8', errors='replace'))
                        target_text_repr = repr(target_bytes.decode('utf-8', errors='replace'))

                        is_verbose_val = CONFIG.verbose_inference 

                        print(f"\n--- Val Sample {i+1}/{len(validation_sequences)} (Prompt Len: {prompt_len}, Gen Len: {response_len_to_generate}) ---")
                        print(f"Prompt: {prompt_text_repr}")
                        print(f"Target: {target_text_repr} (Original suffix for reference)")

                        generated_text_raw = run_conditional_diffusion_inference(
                            model, prompt_ids_val, prompt_mask_val, response_len_to_generate,
                            CONFIG, CONFIG.inference_steps, verbose=is_verbose_val
                        )
                        generated_text_repr = repr(generated_text_raw)

                        print(f"Gen:    {generated_text_repr}")
                        if is_verbose_val:
                            print("-" * 15 + " End Verbose " + "-" * 15)
                    print("="*60 + "\n")
                    model.train()
                else:
                    print(f"\nStep {global_step}: No validation sequences loaded to run prediction.")


        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss_total / total_batches_in_dataset / CONFIG.gradient_accumulation_steps if total_batches_in_dataset > 0 else 0.0
        print(f"\n--- Epoch {epoch+1} Finished ---")
        print(f"Time Taken: {epoch_duration:.2f} seconds")
        print(f"Average Training Loss for Epoch: {avg_epoch_loss:.4f}")
        print(f"Global Step Reached: {global_step}")
        print("---------------------------\n")


    total_training_time = time.time() - overall_start_time
    print("--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    
    if hasattr(train_dataset, 'close'):
        train_dataset.close()
    writer.close()


if __name__ == "__main__":
    log_dir_base = "runs"
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(log_dir_base) / f"{CONFIG.name}_{current_time_str}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs will be written to: {log_dir}")

    train_file = Path(CONFIG.train_data_file)
    if not train_file.exists() or train_file.stat().st_size == 0:
        print(f"Training file '{train_file}' not found or empty. Attempting to create a dummy file.")
    validation_file = Path(CONFIG.validation_data_file)
    if not validation_file.exists() or validation_file.stat().st_size == 0:
        print(f"Validation file '{validation_file}' not found or empty. Attempting to create a dummy file.")
        try:
            val_dummy_text = ("Validation sentence one.\nHas newline.\n" * (CONFIG.max_length // 10))
            if validation_file.parent: 
                validation_file.parent.mkdir(parents=True, exist_ok=True)
            validation_file.write_bytes(val_dummy_text.encode('utf-8', errors='replace') * 5)
            print(f"Dummy validation file created: '{validation_file}'")
        except Exception as e:
            print(f"Error creating dummy validation file '{validation_file}': {e}")

    main()
