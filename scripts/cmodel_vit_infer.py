import os
import sys
import numpy as np
import json
from pathlib import Path

# Provide standard absolute paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATTERNS_TVM_DIR = os.path.join(ROOT_DIR, "patterns_tvm")
GOLDEN_OUT_DIR = os.path.join(ROOT_DIR, "patterns_cmodel_golden")

sys.path.append(ROOT_DIR)
from cmodel_rtl_reference.linear_cmodel_reference import int_dense_kernel, int_matmul_kernel
from cmodel_rtl_reference.nonlinear_cmodel_reference import (
    int_layer_norm_fixed,
    int_gelu_kernel_fixed,
    int_softmax_kernel_fixed
)

def save_npy_and_hex(data, base_path, dtype_hint=None):
    np.save(f"{base_path}.npy", data)
    if dtype_hint is None:
        if data.dtype == np.int8: dtype_hint = 'int8'
        elif data.dtype in [np.int32, np.int64]: dtype_hint = 'int32'
        elif data.dtype == np.int16: dtype_hint = 'int16'
        else: return
        
    with open(f"{base_path}.txt", 'w') as f:
        f.write(f"# Shape: {data.shape}\n")
        f.write(f"# Dtype: {data.dtype}\n")
        flat = data.flatten()
        if dtype_hint == 'int8':
            for i in range(0, len(flat), 16):
                row = flat[i:i+16]
                f.write(" ".join(f"{int(v) & 0xFF:02X}" for v in row) + "\n")
        elif dtype_hint == 'int16':
            for i in range(0, len(flat), 8):
                row = flat[i:i+8]
                f.write(" ".join(f"{int(v) & 0xFFFF:04X}" for v in row) + "\n")
        elif dtype_hint in ['int32', 'int64']:
            for v in flat:
                v_int = int(v) & 0xFFFFFFFF if int(v) < 0 else int(v)
                f.write(f"{v_int:08X}\n")

def get_eff_scale_ms(in_scale, out_scale, max_bit=31):
    eff_scale = in_scale / out_scale
    mantissa, exponent = np.frexp(eff_scale)
    M = np.round(mantissa * (2 ** max_bit)).astype(np.int64)
    S = (max_bit - exponent).astype(np.int64)
    return M, S

class DeiTTinyCModel:
    def __init__(self, tvm_patterns_dir, golden_dir):
        self.tvm_dir = Path(tvm_patterns_dir)
        self.out_dir = Path(golden_dir)
        self.weights_dir = self.tvm_dir / "weights"
        self.scales_dir = self.tvm_dir / "scales"
        
        self.out_golden_dir = self.out_dir / "golden"
        self.out_scales_dir = self.out_dir / "scales"
        
        self.out_golden_dir.mkdir(parents=True, exist_ok=True)
        self.out_scales_dir.mkdir(parents=True, exist_ok=True)
        
        # Load scales cache to avoid loading per inference step
        with open(self.scales_dir / "all_scales.json", "r") as f:
            self.tvm_all_scales = json.load(f)

    def _load_w(self, name):
        return np.load(self.weights_dir / f"{name}.npy")
    
    def _load_s(self, name):
        p = self.scales_dir / f"{name}_float.npy"
        if p.exists():
            return np.load(p).flatten()
        return None

    def _save_mid(self, val, name):
        save_npy_and_hex(val, self.out_golden_dir / f"{name}_int")

    def int_requantize(self, data, in_scale, out_scale, out_dtype, step_name):
        M, S = get_eff_scale_ms(in_scale, out_scale)
        
        # Save M/S for RTL team
        np.save(self.out_scales_dir / f"{step_name}_M.npy", M)
        np.save(self.out_scales_dir / f"{step_name}_S.npy", S)
        
        M = M.flatten().reshape(1, 1, -1) if M.size > 1 else M
        S = S.flatten().reshape(1, 1, -1) if S.size > 1 else S
        
        data_int64 = data.astype(np.int64)
        M_int64 = M.astype(np.int64)
        S_int64 = S.astype(np.int64)
        print(f"[{step_name}] int_requantize: data={data_int64.shape}, M={M_int64.shape}, S={S_int64.shape}")
        
        add_val = np.left_shift(np.int64(1), S_int64 - 1)
        result = np.right_shift((data_int64 * M_int64) + add_val, S_int64)
        
        if out_dtype == np.int8:
            result = np.clip(result, -128, 127).astype(np.int8)
        elif out_dtype == np.int16:
            result = np.clip(result, -32768, 32767).astype(np.int16)
        else:
            result = result.astype(np.int32)
            
        self._save_mid(result, step_name)
        return result

    def int_add(self, lhs, rhs, lhs_s, rhs_s, out_s, step_name):
        # Hardware add requires alignment based on input scales if they are different.
        # But in TVM add, the requantization might be handled specifically. 
        # Actually TVM's `qnn.add` does alignment internally:
        # C = requantize(A) + requantize(B)? 
        # Let's emulate TVM exactly:
        req_lhs, lhs_scale_f, lhs_zp = lhs, lhs_s, 0
        req_rhs, rhs_scale_f, rhs_zp = rhs, rhs_s, 0
        
        # We can implement a float addition emulation, then requantize to out_s.
        # But hardware usually requantizes BOTH to out_s, then adds!
        # Wait, the I-ViT implementation converts inputs to float scale, adds, and scales back.
        # In this C-Model, to match exact HW behavior which might just be:
        # result = requant(lhs, lhs_s, out_s) + requant(rhs, rhs_s, out_s)
        lhs_req = self.int_requantize(lhs, lhs_s, out_s, np.int32, f"_{step_name}_lhs_req")
        rhs_req = self.int_requantize(rhs, rhs_s, out_s, np.int32, f"_{step_name}_rhs_req")
        
        # Add output dtype usually int16 or int32 based on golden patterns.
        # Wait, the golden pattern for `block_0_add1` is int16.
        result = np.clip(lhs_req + rhs_req, -32768, 32767).astype(np.int16)
        self._save_mid(result, step_name)
        return result

    def forward(self, x):
        # 0. Embed
        embed_w = self._load_w("embed_conv_weight")
        embed_bias = self._load_w("embed_conv_bias").flatten()
        
        # Instead of conv2d, since kernel is 16x16, stride 16x16, it is exactly dense on patches.
        # TVM golden `embed_conv` is already flattened.
        # The input `x` has shape (1, 3, 224, 224). We extract 16x16 patches -> (1, 196, 768)
        B, C, H, W = x.shape
        x_patches = x.reshape(B, C, H // 16, 16, W // 16, 16).transpose(0, 2, 4, 1, 3, 5).reshape(B, 196, 3*16*16)
        embed_w_dense = embed_w.reshape(embed_w.shape[0], -1) # (192, 768)
        
        emb = int_dense_kernel(x_patches, embed_w_dense, embed_bias)
        self._save_mid(emb, "embed_conv")
        
        # Requant + Pos
        emb_s0 = self._load_s("qconfig_embed_conv_output_scale")
        emb_add_is = self._load_s("qconfig_addpos_input_scale")
        emb_req = self.int_requantize(emb, emb_s0, emb_add_is, np.int8, "embed_req")
        
        cls_token = self._load_w("cls_token_weight")
        cls_token = np.clip(np.round(cls_token / emb_add_is), -128, 127).astype(np.int8)
        cls_token = np.repeat(cls_token, B, axis=0) # (1, 1, 192)
        emb_concat = np.concatenate([cls_token, emb_req], axis=1)
        
        pos_embed = self._load_w("pos_embed_weight")
        pos_emb_os = self._load_s("qconfig_pos_output_scale")
        pos_embed = np.clip(np.round(pos_embed / pos_emb_os), -128, 127).astype(np.int8)
        emb_add_os = self._load_s("qconfig_addpos_output_scale")
        
        body = self.int_add(emb_concat, pos_embed, emb_add_is, pos_emb_os, emb_add_os, "embed_add_pos")
        
        # 1. Loop over blocks
        for i in range(12):
            body = self.forward_block(body, i)
            
        # 2. Final norm
        norm_b = self._load_w("norm_bias")
        norm = int_layer_norm_fixed(body, norm_b)
        self._save_mid(norm, "final_norm")
        
        # 3. Head
        cls = norm[:, 0:1, :].reshape(B, -1)
        norm_os = self._load_s("qconfig_norm_output_scale")
        head_is = self._load_s("qconfig_head_input_scale")
        cls_req = self.int_requantize(cls, norm_os, head_is, np.int8, "req_norm_to_head")
        
        head_w = self._load_w("head_weight")
        head_b = self._load_w("head_bias").flatten()
        head = int_dense_kernel(cls_req, head_w, head_b)
        self._save_mid(head, "head")
        
        return head
        
    def forward_block(self, body, i):
        # Normalization 1
        norm1_b = self._load_w(f"block_{i}_norm1_bias")
        norm1 = int_layer_norm_fixed(body, norm1_b)
        self._save_mid(norm1, f"block_{i}_norm1")
        
        # Requantize norm1
        norm1_os = self._load_s(f"block_{i}_qconfig_norm1_output_scale")
        qkv_is = self._load_s(f"block_{i}_qconfig_qkv_input_scale")
        req1 = self.int_requantize(norm1, norm1_os, qkv_is, np.int8, f"block_{i}_req_norm1_to_qkv")
        
        # QKV
        qkv_w = self._load_w(f"block_{i}_attn_qkv_weight")
        qkv_b = self._load_w(f"block_{i}_attn_qkv_bias")
        qkv = int_dense_kernel(req1, qkv_w, qkv_b)
        self._save_mid(qkv, f"block_{i}_qkv")
        
        # Requantize for matmul1
        qkv_os = self._load_s(f"block_{i}_qconfig_qkv_output_scale")
        mm1_is = self._load_s(f"block_{i}_qconfig_matmul_1_input_scale")
        req2 = self.int_requantize(qkv, qkv_os, mm1_is, np.int8, f"block_{i}_req_qkv_to_matmul1")
        
        # Split QKV
        B, N, C3 = req2.shape
        C_h = C3 // 3 // 3 # dim per head
        qkv_reshape = req2.reshape(B, N, 3, 3, C_h).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv_reshape[0], qkv_reshape[1], qkv_reshape[2]
        
        # Matmul 1
        attn = int_matmul_kernel(q, k)
        self._save_mid(attn, f"block_{i}_attn_matmul1")
        
        # Softmax
        # Wait, the QK scale in the TVM script was output_scale * qk_scale (1/8 -> 0.125)
        # So the input scale to softmax in float is matmul_1_output_scale * 0.125
        attn_os = self._load_s(f"block_{i}_qconfig_matmul_1_output_scale")
        sm_is = self._load_s(f"block_{i}_qconfig_softmax_input_scale")
        
        req3 = self.int_requantize(attn, attn_os * 0.125, sm_is, np.int8, f"block_{i}_req_attn_to_softmax")
        sm_scale_val = sm_is[0] if getattr(sm_is, 'size', 0) > 1 else (sm_is.item() if hasattr(sm_is, 'item') else sm_is)
        sm_x0_int = int(np.floor(-np.log(2) / float(sm_scale_val)))
        attn_soft = int_softmax_kernel_fixed(req3, sm_x0_int)
        self._save_mid(attn_soft, f"block_{i}_softmax")
        
        # Matmul 2
        sm_r = attn_soft.reshape(B * 3, N, N)
        # Acording to standard attention:
        # attn2 = softmax @ v. Our int_matmul_kernel does y.transpose(0, 1, 3, 2).
        # So we pass v_t to get correctly computed softmax @ v.
        v_transpose = v.transpose(0, 1, 3, 2)
        attn2 = int_matmul_kernel(attn_soft, v_transpose) 
        # reshape back
        attn2 = attn2.transpose(0, 2, 1, 3).reshape(1, 197, 192)
        self._save_mid(attn2, f"block_{i}_attn_matmul2")
        
        # Projection
        mm2_os = self._load_s(f"block_{i}_qconfig_matmul_2_output_scale")
        proj_is = self._load_s(f"block_{i}_qconfig_proj_input_scale")
        req5 = self.int_requantize(attn2, mm2_os, proj_is, np.int8, f"block_{i}_req_attn_to_proj")
        
        proj_w = self._load_w(f"block_{i}_attn_proj_weight")
        proj_b = self._load_w(f"block_{i}_attn_proj_bias")
        proj = int_dense_kernel(req5, proj_w, proj_b)
        self._save_mid(proj, f"block_{i}_proj")
        
        # Add1
        proj_os = self._load_s(f"block_{i}_qconfig_proj_output_scale")
        add1_is = self._load_s(f"block_{i}_qconfig_add1_input_scale")
        # In TVM extract, shortcut scale was qconfig0.input_scale (which is body's prev scale)
        # In our case, the body scale is from prev block's add2 output_scale, OR embed_add_pos output_scale.
        body_s = self._load_s(f"block_{i-1}_qconfig_add2_output_scale") if i > 0 else self._load_s("qconfig_addpos_output_scale")
        add1_os = self._load_s(f"block_{i}_qconfig_add1_output_scale")
        add1 = self.int_add(proj, body, proj_os, body_s, add1_os, f"block_{i}_add1")
        
        # Normalization 2
        norm2_b = self._load_w(f"block_{i}_norm2_bias")
        norm2 = int_layer_norm_fixed(add1, norm2_b)
        self._save_mid(norm2, f"block_{i}_norm2")
        
        # FC1
        norm2_os = self._load_s(f"block_{i}_qconfig_norm2_output_scale")
        fc1_is = self._load_s(f"block_{i}_qconfig_fc1_input_scale")
        req8 = self.int_requantize(norm2, norm2_os, fc1_is, np.int8, f"block_{i}_req_norm2_to_fc1")
        
        fc1_w = self._load_w(f"block_{i}_mlp_fc1_weight")
        fc1_b = self._load_w(f"block_{i}_mlp_fc1_bias")
        fc1 = int_dense_kernel(req8, fc1_w, fc1_b)
        self._save_mid(fc1, f"block_{i}_fc1")
        
        # GELU
        fc1_os = self._load_s(f"block_{i}_qconfig_fc1_output_scale")
        gelu_is = self._load_s(f"block_{i}_qconfig_gelu_input_scale")
        req9 = self.int_requantize(fc1, fc1_os, gelu_is, np.int8, f"block_{i}_req_fc1_to_gelu")
        gelu_scale_val = gelu_is[0] if getattr(gelu_is, 'size', 0) > 1 else (gelu_is.item() if hasattr(gelu_is, 'item') else gelu_is)
        x0_int = int(np.round(-3.0 / float(gelu_scale_val)))
        act = int_gelu_kernel_fixed(req9, x0_int)
        self._save_mid(act, f"block_{i}_gelu")
        
        # FC2
        gelu_os = self._load_s(f"block_{i}_qconfig_gelu_output_scale")
        fc2_is = self._load_s(f"block_{i}_qconfig_fc2_input_scale")
        req10 = self.int_requantize(act, gelu_os, fc2_is, np.int8, f"block_{i}_req_gelu_to_fc2")
        
        fc2_w = self._load_w(f"block_{i}_mlp_fc2_weight")
        fc2_b = self._load_w(f"block_{i}_mlp_fc2_bias")
        fc2 = int_dense_kernel(req10, fc2_w, fc2_b)
        self._save_mid(fc2, f"block_{i}_fc2")
        
        # Add2
        fc2_os = self._load_s(f"block_{i}_qconfig_fc2_output_scale")
        add2_os = self._load_s(f"block_{i}_qconfig_add2_output_scale")
        add2 = self.int_add(fc2, add1, fc2_os, add1_os, add2_os, f"block_{i}_add2")
        
        return add2

if __name__ == "__main__":
    tvm_dir = PATTERNS_TVM_DIR
    out_dir = GOLDEN_OUT_DIR
    
    model = DeiTTinyCModel(tvm_dir, out_dir)
    print("Loading test image input...")
    in_val = np.load(os.path.join(tvm_dir, "input", "input_int8.npy"))
    
    print("Starting End-to-End inference and generating HW Golden Patterns...")
    model.forward(in_val)
    print(f"Extraction complete! Patterns saved in {GOLDEN_OUT_DIR}")
