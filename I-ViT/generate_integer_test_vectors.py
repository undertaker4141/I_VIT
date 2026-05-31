import os
import sys
import numpy as np
import json
from pathlib import Path

# Add paths to import pure_numpy_cmodel and linear/nonlinear reference
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from cmodel_rtl_reference.linear_cmodel_reference import int_dense_kernel, int_matmul_kernel
from cmodel_rtl_reference.nonlinear_cmodel_reference import (
    int_layer_norm_fixed,
    int_gelu_kernel_fixed,
    int_softmax_kernel_fixed
)

def get_eff_scale_ms(in_scale, out_scale, max_bit=31):
    eff_scale = in_scale / out_scale
    mantissa, exponent = np.frexp(eff_scale)
    M = np.round(mantissa * (2 ** max_bit)).astype(np.int64)
    S = (max_bit - exponent).astype(np.int64)
    return M, S

def save_hex_file(data, filename, is_int8=False):
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0: val = (1 << 32) + val
            f.write(f"{val & 0xFFFFFFFF:08x}\n")

def save_hex_file_16(data, filename):
    with open(filename, 'w') as f:
        for val in data.flatten():
            val = int(val)
            if val < 0: val = (1 << 32) + val
            f.write(f"{val & 0xFFFFFFFF:08x}\n")

class ExtractorCModel:
    def __init__(self, tvm_patterns_dir, out_dir):
        self.tvm_dir = Path(tvm_patterns_dir)
        self.out_dir = Path(out_dir)
        self.weights_dir = self.tvm_dir / 'weights'
        self.scales_dir = self.tvm_dir / 'scales'
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Counters
        self.ln_count = 0
        
    def _load_w(self, name):
        return np.load(self.weights_dir / f"{name}.npy")
    
    def _load_s(self, name):
        p = self.scales_dir / f"{name}_float.npy"
        if p.exists():
            return np.load(p).flatten()
        return None

    def int_requantize(self, data, in_scale, out_scale, out_dtype):
        M, S = get_eff_scale_ms(in_scale, out_scale)
        M_arr = M.flatten().reshape(1, 1, -1) if M.size > 1 else M
        S_arr = S.flatten().reshape(1, 1, -1) if S.size > 1 else S
        
        data_int64 = data.astype(np.int64)
        M_int64 = M_arr.astype(np.int64)
        S_int64 = S_arr.astype(np.int64)
        
        add_val = np.left_shift(np.int64(1), S_int64 - 1)
        result = np.right_shift((data_int64 * M_int64) + add_val, S_int64)
        
        if out_dtype == np.int8:
            result = np.clip(result, -128, 127).astype(np.int8)
        elif out_dtype == np.int16:
            result = np.clip(result, -32768, 32767).astype(np.int16)
        else:
            result = result.astype(np.int32)
            
        return result, M, S

    def int_add(self, lhs, rhs, lhs_s, rhs_s, out_s):
        lhs_req, _, _ = self.int_requantize(lhs, lhs_s, out_s, np.int32)
        rhs_req, _, _ = self.int_requantize(rhs, rhs_s, out_s, np.int32)
        result = np.clip(lhs_req + rhs_req, -32768, 32767).astype(np.int16)
        return result

    def forward(self, x):
        print("[1/4] Embedding ...")
        embed_w = self._load_w("embed_conv_weight")
        embed_bias = self._load_w("embed_conv_bias").flatten()
        
        B, C, H, W = x.shape
        x_patches = x.reshape(B, C, H // 16, 16, W // 16, 16).transpose(0, 2, 4, 1, 3, 5).reshape(B, 196, 3*16*16)
        embed_w_dense = embed_w.reshape(embed_w.shape[0], -1)
        
        emb = int_dense_kernel(x_patches, embed_w_dense, embed_bias)
        
        emb_s0 = self._load_s("qconfig_embed_conv_output_scale")
        emb_add_is = self._load_s("qconfig_addpos_input_scale")
        emb_req, _, _ = self.int_requantize(emb, emb_s0, emb_add_is, np.int8)
        
        cls_token = self._load_w("cls_token_weight")
        cls_token = np.clip(np.round(cls_token / emb_add_is), -128, 127).astype(np.int8)
        cls_token = np.repeat(cls_token, B, axis=0)
        emb_concat = np.concatenate([cls_token, emb_req], axis=1)
        
        pos_embed = self._load_w("pos_embed_weight")
        pos_emb_os = self._load_s("qconfig_pos_output_scale")
        pos_embed = np.clip(np.round(pos_embed / pos_emb_os), -128, 127).astype(np.int8)
        emb_add_os = self._load_s("qconfig_addpos_output_scale")
        
        body = self.int_add(emb_concat, pos_embed, emb_add_is, pos_emb_os, emb_add_os)
        
        for i in range(12):
            print(f"[2/4] Processing Block {i} ...")
            body = self.forward_block(body, i)
            
        print("[3/4] Final Norm & Head ...")
        norm_b = self._load_w("norm_bias")
        norm = int_layer_norm_fixed(body, norm_b)
        
        # Save Final Norm
        save_hex_file_16(body, self.out_dir / f"layernorm_input_{self.ln_count}.hex")
        save_hex_file(norm, self.out_dir / f"layernorm_golden_{self.ln_count}.hex")
        save_hex_file(norm_b, self.out_dir / f"layernorm_bias_{self.ln_count}.hex")
        
        norm_os = self._load_s("qconfig_norm_output_scale")
        head_is = self._load_s("qconfig_head_input_scale")
        cls = norm[:, 0:1, :].reshape(B, -1)
        cls_req, M_norm, S_norm = self.int_requantize(cls, norm_os, head_is, np.int8)
        
        # Since cls is just the first token, let's just save M and S
        save_hex_file(M_norm, self.out_dir / f"layernorm_M_{self.ln_count}.hex")
        save_hex_file(S_norm, self.out_dir / f"layernorm_S_{self.ln_count}.hex")
        
        print("[4/4] Done!")
        
    def forward_block(self, body, i):
        # -------------------------------------------------------------
        # 1. Normalization 1
        # -------------------------------------------------------------
        norm1_b = self._load_w(f"block_{i}_norm1_bias")
        norm1 = int_layer_norm_fixed(body, norm1_b)
        
        norm1_os = self._load_s(f"block_{i}_qconfig_norm1_output_scale")
        qkv_is = self._load_s(f"block_{i}_qconfig_qkv_input_scale")
        req1, M_norm1, S_norm1 = self.int_requantize(norm1, norm1_os, qkv_is, np.int8)
        
        save_hex_file_16(body, self.out_dir / f"layernorm_input_{self.ln_count}.hex")
        save_hex_file(norm1, self.out_dir / f"layernorm_golden_{self.ln_count}.hex")
        save_hex_file(req1, self.out_dir / f"layernorm_requant_golden_{self.ln_count}.hex", is_int8=True)
        save_hex_file(norm1_b, self.out_dir / f"layernorm_bias_{self.ln_count}.hex")
        
        M_norm1_full = np.full(192, M_norm1.item()) if M_norm1.size == 1 else M_norm1
        S_norm1_full = np.full(192, S_norm1.item()) if S_norm1.size == 1 else S_norm1
        save_hex_file(M_norm1_full, self.out_dir / f"layernorm_M_{self.ln_count}.hex")
        save_hex_file(S_norm1_full, self.out_dir / f"layernorm_S_{self.ln_count}.hex")
        self.ln_count += 1
        
        # -------------------------------------------------------------
        # 2. QKV & MatMul1 & Softmax
        # -------------------------------------------------------------
        qkv_w = self._load_w(f"block_{i}_attn_qkv_weight")
        qkv_b = self._load_w(f"block_{i}_attn_qkv_bias")
        qkv = int_dense_kernel(req1, qkv_w, qkv_b)
        
        qkv_os = self._load_s(f"block_{i}_qconfig_qkv_output_scale")
        mm1_is = self._load_s(f"block_{i}_qconfig_matmul_1_input_scale")
        req2, _, _ = self.int_requantize(qkv, qkv_os, mm1_is, np.int8)
        
        B, N, C3 = req2.shape
        C_h = C3 // 3 // 3
        qkv_reshape = req2.reshape(B, N, 3, 3, C_h).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv_reshape[0], qkv_reshape[1], qkv_reshape[2]
        
        attn = int_matmul_kernel(q, k)
        
        attn_os = self._load_s(f"block_{i}_qconfig_matmul_1_output_scale")
        sm_is = self._load_s(f"block_{i}_qconfig_softmax_input_scale")
        req3, _, _ = self.int_requantize(attn, attn_os * 0.125, sm_is, np.int8)
        
        sm_scale_val = sm_is[0] if getattr(sm_is, 'size', 0) > 1 else (sm_is.item() if hasattr(sm_is, 'item') else sm_is)
        sm_x0_int = int(np.floor(-np.log(2) / float(sm_scale_val)))
        
        # Save Softmax vectors
        save_hex_file(req3, self.out_dir / f"softmax_input_{i}.hex", is_int8=True)
        save_hex_file(np.array([sm_x0_int], dtype=np.int32), self.out_dir / f"softmax_x0_{i}.hex")
        
        attn_soft = int_softmax_kernel_fixed(req3, sm_x0_int)
        
        save_hex_file(attn_soft, self.out_dir / f"softmax_golden_{i}.hex")
        
        # -------------------------------------------------------------
        # 3. MatMul2 & Proj & Add1
        # -------------------------------------------------------------
        sm_r = attn_soft.reshape(B * 3, N, N)
        v_transpose = v.transpose(0, 1, 3, 2)
        attn2 = int_matmul_kernel(attn_soft, v_transpose) 
        attn2 = attn2.transpose(0, 2, 1, 3).reshape(1, 197, 192)
        
        mm2_os = self._load_s(f"block_{i}_qconfig_matmul_2_output_scale")
        proj_is = self._load_s(f"block_{i}_qconfig_proj_input_scale")
        req5, _, _ = self.int_requantize(attn2, mm2_os, proj_is, np.int8)
        
        proj_w = self._load_w(f"block_{i}_attn_proj_weight")
        proj_b = self._load_w(f"block_{i}_attn_proj_bias")
        proj = int_dense_kernel(req5, proj_w, proj_b)
        
        proj_os = self._load_s(f"block_{i}_qconfig_proj_output_scale")
        body_s = self._load_s(f"block_{i-1}_qconfig_add2_output_scale") if i > 0 else self._load_s("qconfig_addpos_output_scale")
        add1_os = self._load_s(f"block_{i}_qconfig_add1_output_scale")
        add1 = self.int_add(proj, body, proj_os, body_s, add1_os)
        
        # -------------------------------------------------------------
        # 4. Normalization 2
        # -------------------------------------------------------------
        norm2_b = self._load_w(f"block_{i}_norm2_bias")
        norm2 = int_layer_norm_fixed(add1, norm2_b)
        
        norm2_os = self._load_s(f"block_{i}_qconfig_norm2_output_scale")
        fc1_is = self._load_s(f"block_{i}_qconfig_fc1_input_scale")
        req8, M_norm2, S_norm2 = self.int_requantize(norm2, norm2_os, fc1_is, np.int8)
        
        save_hex_file_16(add1, self.out_dir / f"layernorm_input_{self.ln_count}.hex")
        save_hex_file(norm2, self.out_dir / f"layernorm_golden_{self.ln_count}.hex")
        save_hex_file(req8, self.out_dir / f"layernorm_requant_golden_{self.ln_count}.hex", is_int8=True)
        save_hex_file(norm2_b, self.out_dir / f"layernorm_bias_{self.ln_count}.hex")
        
        M_norm2_full = np.full(192, M_norm2.item()) if M_norm2.size == 1 else M_norm2
        S_norm2_full = np.full(192, S_norm2.item()) if S_norm2.size == 1 else S_norm2
        save_hex_file(M_norm2_full, self.out_dir / f"layernorm_M_{self.ln_count}.hex")
        save_hex_file(S_norm2_full, self.out_dir / f"layernorm_S_{self.ln_count}.hex")
        self.ln_count += 1
        
        # -------------------------------------------------------------
        # 5. FC1 & GELU
        # -------------------------------------------------------------
        fc1_w = self._load_w(f"block_{i}_mlp_fc1_weight")
        fc1_b = self._load_w(f"block_{i}_mlp_fc1_bias")
        fc1 = int_dense_kernel(req8, fc1_w, fc1_b)
        
        fc1_os = self._load_s(f"block_{i}_qconfig_fc1_output_scale")
        gelu_is = self._load_s(f"block_{i}_qconfig_gelu_input_scale")
        
        # THIS is the input to GELU!
        req9, _, _ = self.int_requantize(fc1, fc1_os, gelu_is, np.int8)
        save_hex_file(req9, self.out_dir / f"gelu_input_{i}.hex", is_int8=True)
        
        gelu_scale_val = gelu_is[0] if getattr(gelu_is, 'size', 0) > 1 else (gelu_is.item() if hasattr(gelu_is, 'item') else gelu_is)
        x0_int = int(np.round(-3.0 / float(gelu_scale_val)))
        save_hex_file(np.array([x0_int], dtype=np.int32), self.out_dir / f"gelu_x0_{i}.hex")
        
        act = int_gelu_kernel_fixed(req9, x0_int)
        save_hex_file(act, self.out_dir / f"gelu_golden_{i}.hex")
        
        gelu_os = self._load_s(f"block_{i}_qconfig_gelu_output_scale")
        fc2_is = self._load_s(f"block_{i}_qconfig_fc2_input_scale")
        
        req10, M_gelu, S_gelu = self.int_requantize(act, gelu_os, fc2_is, np.int8)
        save_hex_file(req10, self.out_dir / f"gelu_requant_golden_{i}.hex", is_int8=True)
        
        M_gelu_full = np.full(768, M_gelu.item()) if M_gelu.size == 1 else M_gelu
        S_gelu_full = np.full(768, S_gelu.item()) if S_gelu.size == 1 else S_gelu
        save_hex_file(M_gelu_full, self.out_dir / f"gelu_M_{i}.hex")
        save_hex_file(S_gelu_full, self.out_dir / f"gelu_S_{i}.hex")
        
        # -------------------------------------------------------------
        # 6. FC2 & Add2
        # -------------------------------------------------------------
        fc2_w = self._load_w(f"block_{i}_mlp_fc2_weight")
        fc2_b = self._load_w(f"block_{i}_mlp_fc2_bias")
        fc2 = int_dense_kernel(req10, fc2_w, fc2_b)
        
        fc2_os = self._load_s(f"block_{i}_qconfig_fc2_output_scale")
        add2_os = self._load_s(f"block_{i}_qconfig_add2_output_scale")
        add2 = self.int_add(fc2, add1, fc2_os, add1_os, add2_os)
        
        return add2

if __name__ == "__main__":
    print("===========================================================")
    print("使用 Pure Numpy C-Model 生成端到端推論測試向量")
    print("===========================================================")
    
    tvm_dir = os.path.join(ROOT_DIR, 'patterns_tvm')
    out_dir = '/mnt/c/Users/Public/I-ViT_rtl/nonlinear_verification/test_vectors_golden'
    
    os.makedirs(out_dir, exist_ok=True)
    
    model = ExtractorCModel(tvm_dir, out_dir)
    
    print("載入測試圖片輸入 (input_int8.npy)...")
    in_val = np.load(os.path.join(tvm_dir, "input", "input_int8.npy"))
    
    print("開始端到端推論並提取 hex 向量...")
    model.forward(in_val)
    print(f"提取完成！測試向量已儲存至 {out_dir}")
