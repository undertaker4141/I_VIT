"""
Single-image inference verification:
Runs BOTH PyTorch QAT model AND TVM quantized model on the same test image,
then compares predictions.

Key fix: runs a calibration forward on PT model to get the REAL qact2
act_scaling_factor (which differs from the stale checkpoint value).

Usage: python single_image_inference.py --image /path/to/image.jpg
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os, sys, time, argparse

IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

def load_class_names():
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_classes.txt')
    if os.path.exists(cache):
        with open(cache) as f:
            return [l.strip() for l in f.readlines()]
    try:
        import urllib.request
        urllib.request.urlretrieve(IMAGENET_CLASSES_URL, cache)
        with open(cache) as f:
            return [l.strip() for l in f.readlines()]
    except Exception:
        return [str(i) for i in range(1000)]


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


def float_to_int8(x_f32, input_scale):
    """Quantize float image to int8 for TVM."""
    return np.round(x_f32.numpy() / input_scale).clip(-128, 127).astype('int8')


def top5_from_logits(logits_1d, class_names):
    """Return top-5 (name, prob, class_idx) from raw logits (applies softmax)."""
    probs = F.softmax(torch.tensor(logits_1d), dim=-1).numpy()
    idx   = np.argsort(probs)[::-1][:5]
    return [(class_names[i], float(probs[i]), int(i)) for i in idx]


def top5_from_probs(probs_1d, class_names):
    """Return top-5 (name, prob, class_idx) from probability array (already softmax-ed)."""
    idx = np.argsort(probs_1d)[::-1][:5]
    return [(class_names[i], float(probs_1d[i]), int(i)) for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Absolute path to test image')
    parser.add_argument('--skip-pytorch', action='store_true')
    args = parser.parse_args()

    cwd         = os.getcwd()                              # TVM_benchmark/
    model_path  = os.path.join(cwd, '../checkpoints/qat_calibrated.pth')
    params_path = os.path.join(cwd, 'params.npy')

    # ── 1. Load PyTorch model ─────────────────────────────────────────────────
    print("\n[Step 1] Loading PyTorch QAT model...")
    os.chdir('..')
    sys.path.insert(0, os.getcwd())     # I-ViT/ → models.vit_quant
    from models.vit_quant import deit_tiny_patch16_224

    checkpoint  = torch.load(model_path, map_location='cpu', weights_only=False)
    model_dict  = checkpoint['model'] if 'model' in checkpoint else checkpoint
    pt_model    = deit_tiny_patch16_224(pretrained=False)
    ms          = pt_model.state_dict()
    ld          = {k: v for k, v in model_dict.items()
                   if k in ms and getattr(v, 'shape', None) == ms[k].shape}
    pt_model.load_state_dict(ld, strict=False)
    pt_model.eval()
    print("  PT model loaded.")

    # ── 2. Calibration forward → get REAL act scaling factors ─────────────────
    print("\n[Step 2] Running calibration forward to get real scaling factors...")
    x_f32 = preprocess_float(args.image)

    calib_scales = {}
    def hook_qact2(mod, inp, out):
        calib_scales['qact2.act_scaling_factor'] = float(out[1].flatten()[0])
    def hook_qact_input(mod, inp, out):
        calib_scales['qact_input.act_scaling_factor'] = float(out[1].flatten()[0])

    pt_model.qact2.register_forward_hook(hook_qact2)
    pt_model.qact_input.register_forward_hook(hook_qact_input)

    with torch.no_grad():
        pt_raw = pt_model(x_f32)   # [1, 1000] raw logits

    input_scale   = calib_scales['qact_input.act_scaling_factor']
    qact2_real_sf = calib_scales['qact2.act_scaling_factor']
    qact2_stale   = float(model_dict.get('qact2.act_scaling_factor',
                                          torch.tensor([qact2_real_sf])).flatten()[0])

    print(f"  input_scale      = {input_scale:.8f}")
    print(f"  qact2_sf (stale) = {qact2_stale:.8f}")
    print(f"  qact2_sf (real ) = {qact2_real_sf:.8f}")
    if abs(qact2_stale - qact2_real_sf) / qact2_real_sf > 0.01:
        print(f"  ⚠️  qact2 scale differs by {abs(qact2_stale-qact2_real_sf)/qact2_real_sf*100:.1f}% — using real value for TVM")

    # ── 3. PyTorch inference result ───────────────────────────────────────────
    class_names = load_class_names()
    pt_top5 = None

    if not args.skip_pytorch:
        pt_logits = pt_raw[0].numpy() if isinstance(pt_raw, torch.Tensor) else pt_raw.numpy()[0]
        pt_top5   = top5_from_logits(pt_logits, class_names)
        print("\n  PyTorch QAT Top-5 (after softmax):")
        for rank, (name, prob, idx) in enumerate(pt_top5):
            print(f"    #{rank+1}  [{idx:4d}]  {prob*100:6.2f}%  {name}")

    # ── 4. Switch sys.path back to TVM_benchmark/ ─────────────────────────────
    os.chdir(cwd)
    sys.path.pop(0)
    for k in [k for k in sys.modules if k.startswith('models')]:
        del sys.modules[k]

    # ── 5. Build & run TVM model with REAL scales ─────────────────────────────
    print("\n[Step 3] Building TVM model with calibrated scales...")
    import tvm
    from tvm import relay
    import models.layers as layers
    import models.build_model as build_model
    import convert_model

    # Use auto-loaded calibrated scales from calibrated_scales.npy
    # (The file contains ALL 137 scaling factors, not just qact2)
    convert_model.load_qconfig(model_dict, 12)

    func, _ = build_model.get_workload(
        name='deit_tiny_patch16_224',
        batch_size=1, image_shape=(3, 224, 224),
        dtype='int8', data_layout='NCHW', kernel_layout='OIHW',
    )

    pretrained_params = np.load(params_path, allow_pickle=True)[()]
    x_int8 = float_to_int8(x_f32, input_scale)

    t0 = time.time()
    print("  Compiling... (30-90s)")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target='llvm', params={**pretrained_params})
    print(f"  Compiled in {time.time()-t0:.1f}s")

    dev     = tvm.device('llvm', 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    runtime.set_input('data', x_int8)

    t0 = time.time()
    runtime.run()
    print(f"  Inference in {(time.time()-t0)*1000:.1f}ms")

    tvm_probs = runtime.get_output(0).numpy()[0]  # already softmax-ed
    tvm_top5  = top5_from_probs(tvm_probs, class_names)

    print("\n  TVM Top-5:")
    for rank, (name, prob, idx) in enumerate(tvm_top5):
        print(f"    #{rank+1}  [{idx:4d}]  {prob*100:6.2f}%  {name}")

    # ── 6. Compare ────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  COMPARISON")
    print("="*55)
    tvm_name, tvm_prob, tvm_idx = tvm_top5[0]
    print(f"  TVM  Top-1: [{tvm_idx:4d}] {tvm_name}  ({tvm_prob*100:.2f}%)")

    if pt_top5:
        pt_name, pt_prob, pt_idx = pt_top5[0]
        print(f"  PT   Top-1: [{pt_idx:4d}] {pt_name}  ({pt_prob*100:.2f}%)")
        agree = (pt_idx == tvm_idx)
        status = "✅ AGREE" if agree else "❌ DISAGREE"
        print(f"\n  {status} — PT and TVM predict {'same' if agree else 'different'} class")
        if not agree:
            pt_idxs = [x[2] for x in pt_top5]
            if tvm_idx in pt_idxs:
                print(f"  (TVM's prediction is PT rank #{pt_idxs.index(tvm_idx)+1})")
    print("="*55)


if __name__ == '__main__':
    main()
