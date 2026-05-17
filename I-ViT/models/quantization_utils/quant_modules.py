import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Parameter

from .quant_utils import *


class QuantLinear(nn.Linear):
    """
    Class to quantize weights of given Linear layer

    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 per_channel=True,
                 quant_mode='symmetric'):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        if self.bias is not None:
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception('For weight, we only support per_channel quantization.')

            self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.fc_scaling_factor, True)

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        if self.bias is not None:
            self.bias_integer = self.weight_function(
                self.bias, self.bias_bit, bias_scaling_factor, True)
        else:
            self.bias_integer = None

        # 🔥 MODIFIED: Use integer-only computation to match TVM qnn.dense
        # This ensures training and inference use the same numerical computation
        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        
        # 確保 bias_scaling_factor 在正確的設備上
        bias_scaling_factor = bias_scaling_factor.to(x.device)
        
        # Quantize input to int8 (simulate TVM behavior)
        x_int8 = torch.round(x / prev_act_scaling_factor).clamp(-128, 127)
        
        # Integer matrix multiplication (int8 @ int8 -> int32)
        # Use int32 to avoid overflow during matmul
        # Note: CUDA doesn't support int32 matmul, so we use float and convert back
        x_int32 = x_int8.to(torch.int32)
        weight_int32 = self.weight_integer.to(torch.int32)
        
        # Reshape for batch processing
        original_shape = x_int32.shape
        if len(original_shape) > 2:
            x_int32 = x_int32.reshape(-1, original_shape[-1])
        
        # Integer matmul: [batch, in_features] @ [out_features, in_features].T
        # CUDA doesn't support int32 matmul, so convert to float, compute, then back to int32
        output_int32 = torch.matmul(x_int32.float(), weight_int32.t().float()).round().to(torch.int32)
        
        # Add bias (int32 + int32)
        if self.bias_integer is not None:
            output_int32 = output_int32 + self.bias_integer.to(torch.int32)
        
        # Reshape back if needed
        if len(original_shape) > 2:
            output_int32 = output_int32.reshape(*original_shape[:-1], -1)
        
        # Dequantize: int32 -> float32
        # Use .detach() for integer operations, then allow gradients for scaling
        output = output_int32.detach().float() * bias_scaling_factor
        
        # For backward pass, we need to use Straight-Through Estimator (STE)
        # The gradient flows through as if we did floating-point computation
        if self.training:
            # STE: forward uses integer, backward uses float approximation
            x_float = x / prev_act_scaling_factor
            with torch.no_grad():
                output_float = F.linear(x_float, weight=self.weight_integer.float(), 
                                       bias=self.bias_integer.float() if self.bias_integer is not None else None) \
                              * bias_scaling_factor
            # Replace forward value with integer result, keep float gradient
            output = output_int32.float() * bias_scaling_factor + (output_float - output_float.detach())
        
        return output, bias_scaling_factor


class QuantAct(nn.Module):
    """
    Class to quantize given activations
    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'asymmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """

    def __init__(self,
                 activation_bit=8,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 quant_mode="symmetric"):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.per_channel = per_channel

        self.min_val = torch.zeros(1)
        self.max_val = torch.zeros(1)
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode: {2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(), self.x_max.item())

    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x,
                pre_act_scaling_factor=None,
                identity=None,
                identity_scaling_factor=None):
        # collect runnng stats
        with torch.no_grad():
            x_act = x if identity is None else identity + x
            if self.running_stat:
                if len(x_act.shape) == 4:
                    x_act = x_act.permute(0, 2, 3, 1)
                v = x_act.reshape(-1, x_act.shape[-1])
                v = v.transpose(0, 1)

                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                if torch.eq(self.min_val, self.max_val).all():
                    self.min_val = cur_min
                    self.max_val = cur_max
                else:
                    self.min_val = self.min_val * self.act_range_momentum + \
                                 cur_min * (1 - self.act_range_momentum)
                    self.max_val = self.max_val * self.act_range_momentum + \
                                   cur_max * (1 - self.act_range_momentum)
                self.max_val = self.max_val.max()
                self.min_val = self.min_val.min()

            self.act_scaling_factor = symmetric_linear_quantization_params(
                self.activation_bit, self.min_val, self.max_val)

        if pre_act_scaling_factor is None:
            # this is for the input quantization
            quant_act_int = self.act_function(x, self.activation_bit, self.act_scaling_factor, False)
        else:
            quant_act_int = fixedpoint_mul.apply(
                x, pre_act_scaling_factor,
                self.activation_bit, self.quant_mode,
                self.act_scaling_factor,
                identity, identity_scaling_factor)

        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given matmul layer
    """
    def __init__(self):
        super(QuantMatMul, self).__init__()
        self.register_buffer('act_scaling_factor', torch.zeros(1))
    
    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        A_int = A / pre_act_scaling_factor_A
        B_int = B / pre_act_scaling_factor_B
        act_scaling_factor = pre_act_scaling_factor_A * pre_act_scaling_factor_B
        self.act_scaling_factor = act_scaling_factor
        return (A_int @ B_int) * act_scaling_factor, act_scaling_factor


class QuantConv2d(nn.Conv2d):
    """
    Class to quantize weights of given convolutional layer
    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weight_bit=8,
                 bias_bit=32,
                 quant_mode="symmetric",
                 per_channel=True,
                 weight_percentile=0):
        super(QuantConv2d, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias
                                          )
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)

        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(self.weight_bit, self.quant_mode)
        return s

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, pre_act_scaling_factor=None):
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        with torch.no_grad():
            w = self.weight
            if self.per_channel:
                v = w.reshape(w.shape[0], -1)
                cur_min = v.min(axis=1).values
                cur_max = v.max(axis=1).values
                self.min_val = cur_min
                self.max_val = cur_max
            else:
                raise Exception('For weight, we only support per_channel quantization.')

            self.conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, self.min_val, self.max_val)

        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.conv_scaling_factor, True)
        bias_scaling_factor = self.conv_scaling_factor * pre_act_scaling_factor
        self.bias_integer = self.weight_function(
            self.bias, self.bias_bit, bias_scaling_factor, True)

        # 🔥 MODIFIED: Use integer-only computation to match TVM qnn.conv2d
        pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
        
        # 確保 bias_scaling_factor 在正確的設備上
        bias_scaling_factor = bias_scaling_factor.to(x.device)
        
        # Quantize input to int8
        x_int8 = torch.round(x / pre_act_scaling_factor).clamp(-128, 127)
        
        # Integer convolution (int8 @ int8 -> int32)
        x_int8_for_conv = x_int8.to(torch.int8)
        weight_int8 = self.weight_integer.to(torch.int8)
        bias_int32 = self.bias_integer.to(torch.int32) if self.bias_integer is not None else None
        
        # PyTorch conv2d with int8 inputs will automatically promote to int32 output
        output_int32 = F.conv2d(x_int8_for_conv.float(), weight_int8.float(), None,
                               self.stride, self.padding, self.dilation, self.groups)
        output_int32 = torch.round(output_int32).to(torch.int32)
        
        # Add bias
        if bias_int32 is not None:
            output_int32 = output_int32 + bias_int32.view(1, -1, 1, 1)
        
        # Dequantize
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)
        output = output_int32.detach().float() * correct_output_scale
        
        # STE for training
        if self.training:
            x_float = x / pre_act_scaling_factor
            with torch.no_grad():
                output_float = F.conv2d(x_float, self.weight_integer.float(), 
                                       self.bias_integer.float() if self.bias_integer is not None else None,
                                       self.stride, self.padding, self.dilation, self.groups) * correct_output_scale
            output = output_int32.float() * correct_output_scale + (output_float - output_float.detach())
        
        return output, correct_output_scale


class IntLayerNorm(nn.LayerNorm):
    """
    Implementation of I-LayerNorm
    Class to quantize given LayerNorm layer

    修改說明 (2026-01-09):
    - 新增 output_integer buffer 保存內部的整數計算結果 (y_int + bias_int)
    - 這個值可用於 C-model 驗證，不包含 per-channel scaling_factor 的浮點誤差
    """
    def __init__(self, 
                normalized_shape, 
                eps=1e-5,
                elementwise_affine=True):
        super(IntLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.dim_sqrt = None
        self.register_buffer('norm_scaling_factor', torch.zeros(1))
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))
        # 新增：保存內部整數輸出，用於 C-model 驗證
        self.register_buffer('output_integer', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, scaling_factor=None):
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float, device=x.device)
            self.dim_sqrt = torch.sqrt(n)

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_sq_int = y_int ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # Integer Iteration
        k = 2 ** 16
        for _ in range(10):
            k_1 = floor_ste.apply((k + floor_ste.apply(var_int/k))/2)
            k = k_1
        std_int = k

        factor = floor_ste.apply((2 ** 31-1) / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2 ** 30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        self.bias_integer = bias_int

        y_int = y_int + bias_int

        # 保存內部整數輸出 (這是真正的整數計算結果，供 C-model 驗證)
        self.output_integer = y_int.detach()
        
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor
        self.norm_scaling_factor = scaling_factor
        return x, scaling_factor


class IntGELU(nn.Module):
    """
    Implementation of ShiftGELU
    Class to quantize given GELU layer
    """

    def __init__(self, output_bit=8):
        super(IntGELU, self).__init__()
        self.output_bit = output_bit

        self.n = 23  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n

        return exp_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        pre_x_int = x / scaling_factor
        scaling_factor_sig = scaling_factor * 1.702

        x_int_max, _ = pre_x_int.max(dim=-1, keepdim=True)
        x_int = pre_x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor_sig) # e^(x-x_max)

        exp_int_max, _ = self.int_exp_shift(-x_int_max, scaling_factor_sig)  # e^(-x_max)
        exp_int_sum = exp_int + exp_int_max

        exp_int_sum.clamp_max_(2**31-1)
        factor = floor_ste.apply((2 ** 31-1) / exp_int_sum)
        sigmoid_int = floor_ste.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        sigmoid_scaling_factor = torch.tensor([1 / 2 ** (self.output_bit-1)], device=x.device, dtype=x.dtype)

        x_int = pre_x_int * sigmoid_int
        scaling_factor = scaling_factor * sigmoid_scaling_factor
        self.act_scaling_factor = scaling_factor
        return x_int * scaling_factor, scaling_factor


class IntSoftmax(nn.Module):
    """
    Implementation of Shiftmax
    Class to quantize given Softmax layer
    """

    def __init__(self, output_bit=8):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit

        self.n = 15  # sufficiently large integer
        #The minimum value for ensuring accuracy (varies depending on models)

        self.register_buffer('act_scaling_factor', torch.zeros(1))

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_exp_shift(self, x_int, scaling_factor):
        x_int = x_int + floor_ste.apply(x_int / 2) - floor_ste.apply(x_int / 2 ** 4)

        with torch.no_grad():
            x0_int = torch.floor(-1.0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int = r/2 - x0_int
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        x_int = x / scaling_factor
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        exp_int, _ = self.int_exp_shift(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        exp_int_sum.clamp_max_(2**31-1)
        factor = floor_ste.apply((2**31-1) / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (31-self.output_bit+1))
        scaling_factor = torch.tensor([1 / 2 ** (self.output_bit-1)], device=x.device, dtype=x.dtype)

        self.act_scaling_factor = scaling_factor
        return exp_int * scaling_factor, scaling_factor
