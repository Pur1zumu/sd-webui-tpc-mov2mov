import math
import sys
import traceback
import psutil

import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

from modules import shared, errors, devices
from modules.hypernetworks import hypernetwork
from modules.sd_hijack_optimizations import get_available_vram
from collections import defaultdict


class KVMem:
    def __init__(self) -> None:
        self._kv_mem = defaultdict(list)
        self.is_new_img = True
        self.cur_img_idx = 0
        self.internal_step = 0
        self.max_mem_size = 1
        self.func_hacked = False
        self.retain_first = True


    def __call__(self, k, v, skip=False):
        if skip or (self.max_mem_size == 0 and not self.retain_first):
            return k, v
        
        self._kv_mem[self.cur_img_idx].append((k.clone().detach().cpu(), v.clone().detach().cpu()))
        if self.is_new_img:
            self.is_new_img = False
            self.internal_step = 0

        if self.cur_img_idx > 0:
            mem_k, mem_v = [], []
            for i in range(max(self.cur_img_idx - self.max_mem_size, 0), self.cur_img_idx):
                _k, _v = self._kv_mem[i][self.internal_step]
                mem_k.append(_k)
                mem_v.append(_v)
            mem_k = torch.cat(mem_k, dim=1).to(k.device)
            mem_v = torch.cat(mem_v, dim=1).to(v.device)
            # k in cuda, mem_k in cpu
            k = torch.cat([mem_k, k], dim=1)
            v = torch.cat([mem_v, v], dim=1)

            mem_k.cpu()
            mem_v.cpu()
            mem_k.detach_()
            mem_v.detach_()

            del _k, _v, mem_k, mem_v
        
        self.internal_step += 1

        return k, v
    
    def reset(self):
        if len(self._kv_mem) > 0:
            for k in list(self._kv_mem.keys()):
                for t, _ in self._kv_mem[k]:
                    t.detach_()
                del self._kv_mem[k]

        self._kv_mem = defaultdict(list)
        self.cur_img_idx = 0
        self.internal_step = 0
        self.is_new_img = True
    
    def iter_reset(self, img_idx):
        self.cur_img_idx = img_idx
        self.is_new_img = True

    def clean_unused_mem(self):
        for k in list(self._kv_mem.keys()):
            if self.retain_first and k == 0:
                continue
            if k < self.cur_img_idx - self.max_mem_size:
                for t, _ in self._kv_mem[k]:
                    t.detach_()
                del self._kv_mem[k]

kv_mem = KVMem()


# taken from https://github.com/Doggettx/stable-diffusion and modified
def split_cross_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    skip = context is not None
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    k_in, v_in = kv_mem(k_in, v_in, skip=skip)

    dtype = q_in.dtype
    if shared.opts.upcast_attn:
        q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k_in = k_in * self.scale
    
        del context, x
    
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in
    
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    
        mem_free_total = get_available_vram()
    
        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1
    
        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")
    
        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')
    
        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)
    
            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2
    
        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)