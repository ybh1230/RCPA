import torch, os
import torch.nn.functional as F
from einops import rearrange


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False


def initialize_attention_priority():
    if os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') is not None:
        return os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION').lower()
    elif FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    elif FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    elif SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    elif XFORMERS_AVAILABLE:
        return "xformers"
    else:
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", required_in_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def _tuple3(value, default):
    value = default if value is None else value
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 2:
        return (1, int(value[0]), int(value[1]))
    return tuple(int(i) for i in value)


def _grid_from_dims(dims, sequence_length):
    dims = {} if dims is None else dims
    frames = dims.get("f", dims.get("frames", dims.get("num_frames", 1)))
    height = dims.get("h", dims.get("height", None))
    width = dims.get("w", dims.get("width", None))
    if height is None or width is None:
        return None
    frames, height, width = int(frames), int(height), int(width)
    if frames * height * width != sequence_length:
        return None
    return frames, height, width


def _pool_sequence_tokens(x, grid, pool_size):
    b, heads, seq_len, dim = x.shape
    frames, height, width = grid
    x = x.reshape(b, heads, frames, height, width, dim)
    x = x.permute(0, 1, 5, 2, 3, 4).reshape(b * heads, dim, frames, height, width)
    pooled = F.avg_pool3d(x, kernel_size=pool_size, stride=pool_size, ceil_mode=True)
    pooled = pooled.reshape(b, heads, dim, -1).permute(0, 1, 3, 2).contiguous()
    return pooled


def _regional_local_mask(grid, window_size, num_memory_tokens, device):
    frames, height, width = grid
    wt, wh, ww = window_size
    coords_t = torch.arange(frames, device=device).view(frames, 1, 1).expand(frames, height, width).reshape(-1)
    coords_y = torch.arange(height, device=device).view(1, height, 1).expand(frames, height, width).reshape(-1)
    coords_x = torch.arange(width, device=device).view(1, 1, width).expand(frames, height, width).reshape(-1)
    local = (
        (coords_t[:, None] - coords_t[None, :]).abs() <= wt // 2
    ) & (
        (coords_y[:, None] - coords_y[None, :]).abs() <= wh // 2
    ) & (
        (coords_x[:, None] - coords_x[None, :]).abs() <= ww // 2
    )
    if num_memory_tokens > 0:
        memory = torch.ones(local.shape[0], num_memory_tokens, dtype=torch.bool, device=device)
        local = torch.cat([local, memory], dim=1)
    return local.view(1, 1, local.shape[0], local.shape[1])


def regional_context_exchange_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pattern="b n s d",
    k_pattern="b n s d",
    v_pattern="b n s d",
    out_pattern="b n s d",
    dims=None,
    scale=None,
):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    grid = _grid_from_dims(dims, q.shape[2])
    if grid is None:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        return rearrange_out(out, out_pattern, required_out_pattern, dims)

    dims = {} if dims is None else dims
    pool_size = _tuple3(dims.get("regional_pool_size", dims.get("pool_size", (1, 4, 4))), (1, 4, 4))
    window_size = _tuple3(dims.get("regional_window_size", dims.get("window_size", (1, 16, 16))), (1, 16, 16))
    exchange_steps = max(0, int(dims.get("regional_exchange_steps", 1)))
    memory_blend = float(dims.get("regional_memory_blend", 0.5))

    q_memory = _pool_sequence_tokens(q, grid, pool_size)
    k_memory = _pool_sequence_tokens(k, grid, pool_size)
    v_memory = _pool_sequence_tokens(v, grid, pool_size)
    for _ in range(exchange_steps):
        exchanged = torch.nn.functional.scaled_dot_product_attention(
            q_memory,
            k_memory,
            v_memory,
            scale=scale,
        )
        v_memory = (1.0 - memory_blend) * v_memory + memory_blend * exchanged
        k_memory = (1.0 - memory_blend) * k_memory + memory_blend * q_memory

    k_augmented = torch.cat([k, k_memory], dim=2)
    v_augmented = torch.cat([v, v_memory], dim=2)
    mask = _regional_local_mask(grid, window_size, k_memory.shape[2], q.device)
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k_augmented,
        v_augmented,
        attn_mask=mask,
        scale=scale,
    )
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def xformers_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, compatibility_mode=False):
    if dims is not None and dims.get("regional_context_exchange", False):
        return regional_context_exchange_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if compatibility_mode or (attn_mask is not None):
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale)
    else:
        if ATTENTION_IMPLEMENTATION == "flash_attention_3":
            return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "flash_attention_2":
            return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "sage_attention":
            return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "xformers":
            return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        else:
            return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
