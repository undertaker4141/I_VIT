"""
Generate a complete set of calibrated scaling factors by running a forward pass.
This fixes the stale checkpoint values that cause TVM inference to fail.

Usage: python generate_calibrated_scales.py --image <path> --output calibrated_scales.npy
"""
import numpy as np
import torch
import os, sys, argparse
from torchvision import transforms
from PIL import Image


def preprocess_float(img_path):
    """Standard ImageNet preprocess → float32 for PyTorch."""
    img = Image.open(img_path).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to calibration image')
    parser.add_argument('--model-path', default='../checkpoints/qat_calibrated.pth',
                        help='Path to QAT checkpoint')
    parser.add_argument('--output', default='calibrated_scales.npy',
                        help='Output file for calibrated scales')
    args = parser.parse_args()
    
    # Save current directory and resolve ALL paths BEFORE changing directories
    original_cwd = os.getcwd()
    output_path = os.path.abspath(args.output)
    image_path = os.path.abspath(args.image)  # 🔥 FIX: Resolve image path before chdir
    model_path = os.path.abspath(args.model_path)  # Also resolve model path
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Load PyTorch model
    os.chdir('..')
    sys.path.insert(0, os.getcwd())
    from models.vit_quant import deit_tiny_patch16_224
    
    pt_model = deit_tiny_patch16_224(pretrained=False)
    ms = pt_model.state_dict()
    ld = {k: v for k, v in model_dict.items()
          if k in ms and getattr(v, 'shape', None) == ms[k].shape}
    pt_model.load_state_dict(ld, strict=False)
    pt_model.eval()
    
    print(f"[1/3] Loading model from {model_path}")
    print(f"[2/3] Running calibration forward on {image_path}")
    
    # Collect all scaling factors during forward pass
    runtime_scales = {}
    
    def make_hook(module_name):
        def hook(mod, inp, out):
            # QuantAct modules return (quantized_output, scale_tensor)
            if isinstance(out, tuple) and len(out) > 1:
                scale = out[1]
                if hasattr(scale, 'flatten'):
                    scale_val = float(scale.flatten()[0])
                else:
                    scale_val = float(scale)
                runtime_scales[module_name + '.act_scaling_factor'] = scale_val
        return hook
    
    # Register hooks on all QuantAct modules
    for name, module in pt_model.named_modules():
        module_type = type(module).__name__
        if 'QuantAct' in module_type or 'qact' in name.lower():
            module.register_forward_hook(make_hook(name))
    
    # Run forward pass (use resolved absolute path)
    x_f32 = preprocess_float(image_path)
    with torch.no_grad():
        _ = pt_model(x_f32)
    
    print(f"[3/3] Captured {len(runtime_scales)} runtime scaling factors")
    
    # Restore original directory before saving
    os.chdir(original_cwd)
    
    # Save to file
    np.save(output_path, runtime_scales)
    print(f"✅ Saved calibrated scales to {output_path}")
    
    # Verify file was created
    if os.path.exists(output_path):
        print(f"✅ File verified at {output_path}")
    else:
        print(f"❌ ERROR: File not found at {output_path}")
    
    # Show comparison with checkpoint
    checkpoint_scales = {}
    for key, tensor in model_dict.items():
        if 'scaling_factor' in key:
            val = tensor.cpu().numpy().reshape((-1))
            if val.size == 1:
                val = float(val[0])
            checkpoint_scales[key] = val
    
    mismatches = 0
    for key in runtime_scales:
        if key in checkpoint_scales:
            ckpt_val = checkpoint_scales[key]
            runtime_val = runtime_scales[key]
            if isinstance(ckpt_val, np.ndarray):
                ckpt_val = float(ckpt_val[0]) if ckpt_val.size == 1 else ckpt_val
            
            diff_pct = abs(ckpt_val - runtime_val) / runtime_val * 100 if runtime_val != 0 else 0
            if diff_pct >= 5.0:
                mismatches += 1
    
    print(f"\n📊 Statistics:")
    print(f"   - Total scaling factors: {len(runtime_scales)}")
    print(f"   - Mismatches >5%: {mismatches}")
    print(f"   - Match rate: {(len(runtime_scales)-mismatches)/len(runtime_scales)*100:.1f}%")


if __name__ == '__main__':
    main()
