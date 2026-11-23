"""Merge the pi-H3 2-step adapter into the BF16 MiniMax H3 FL2VA model and quantize once to INT8 ConvRot.

The official INT8 ConvRot model is used as the template: exactly its INT8 layers are quantized again
(per-row, ConvRot, group 256, the same settings as its comfy_quant metadata) and every other tensor is
copied from it. LoRA layers get W_bf16 + B @ A before quantization; the adapter's non-LoRA tensors
replace the template tensors, as Load LakonLab Adapter does at runtime.

Run it with the Python that belongs to ComfyUI (convert_fused.bat does this). Needs CUDA, comfy-kitchen
from ComfyUI and ComfyUI-piFlow (for its key map). Tensors are read with plain file reads (no memory mapping)
and written one by one, so the process holds about one layer at a time instead of mapping the 40 GB and
21 GB inputs into memory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import urllib.request

import torch

BF16_NAME = "minimax_h3_fl2va_pruned_bf16.safetensors"
BF16_URL = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/" + BF16_NAME
BF16_SIZE = 40225724176
BF16_SHA256 = "a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4"
TEMPLATE_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
ADAPTER_NAME = "pi_h3_2step_adapter.safetensors"
OUTPUT_NAME = "minimax_h3_fl2va_pruned_pih3_2step_fused_int8_convrot.safetensors"
H3_CONFIG = dict(num_layers=50, token_refiner_num_layers=2, num_attention_heads=56, attention_head_dim=128)
QUANT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
LORA_A, LORA_B = ".lora_A.weight", ".lora_B.weight"
GB = 1000 ** 3


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I8": torch.int8, "U8": torch.uint8}


class TensorFile:
    """Read single tensors from a .safetensors file without mapping the whole file into the process."""

    def __init__(self, path):
        self.handle = open(path, "rb", buffering=0)
        size = struct.unpack("<Q", self.handle.read(8))[0]
        self.header = json.loads(self.handle.read(size))
        self.metadata = self.header.pop("__metadata__", None) or {}
        self.base = 8 + size

    def get(self, key):
        info = self.header[key]
        start, end = info["data_offsets"]
        data = bytearray(end - start)
        self.handle.seek(self.base + start)
        view, done = memoryview(data), 0
        while done < len(data):
            read = self.handle.readinto(view[done:])
            if not read:
                raise EOFError(f"{key}: file is truncated")
            done += read
        if not data:
            return torch.empty(info["shape"], dtype=DTYPES[info["dtype"]])
        return torch.frombuffer(data, dtype=DTYPES[info["dtype"]]).reshape(info["shape"])

    def close(self):
        self.handle.close()


def load_piflow_key_map(comfy_root):
    for package in sorted((comfy_root / "custom_nodes").iterdir()):
        if (package / "lakonlab_loaders.py").is_file():
            break
    else:
        raise RuntimeError("ComfyUI-piFlow was not found in custom_nodes; run setup.bat --install-dependencies first")
    sys.path.insert(0, str(comfy_root))
    spec = importlib.util.spec_from_file_location("comfyui_piflow", package / "__init__.py", submodule_search_locations=[str(package)])
    sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[spec.name])
    from comfyui_piflow.lakonlab_loaders import minimax_h3_to_diffusers
    return minimax_h3_to_diffusers(H3_CONFIG)


def model_paths(comfy_root, extra):
    import folder_paths
    from utils.extra_config import load_extra_path_config
    for config in (comfy_root / "extra_model_paths.yaml", extra):
        if config and Path(config).is_file():
            load_extra_path_config(str(config))
    return folder_paths


def find_model(folder_paths, folder, name):
    candidates = folder_paths.get_filename_list(folder)
    matches = [name] if name in candidates else [item for item in candidates if Path(item).name == name]
    if len(matches) != 1:
        raise FileNotFoundError(f"{name} {'is missing from' if not matches else 'has several copies in'} models/{folder}; pass its path explicitly")
    return Path(folder_paths.get_full_path_or_raise(folder, matches[0]))


def download_bf16(directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / BF16_NAME
    if target.is_file() and target.stat().st_size == BF16_SIZE:
        print("BF16 model already downloaded:", target, flush=True)
    else:
        need = BF16_SIZE - (target.stat().st_size if target.is_file() else 0)
        if shutil.disk_usage(directory).free < need + GB:
            raise RuntimeError(f"Not enough free space in {directory}: {need / GB:.1f} GB needed for the BF16 model")
        partial = target.stat().st_size if target.is_file() else 0
        request = urllib.request.Request(BF16_URL, headers={"User-Agent": "H3-piFlow-X2-Stream-convert", "Range": f"bytes={partial}-"})
        print(f"Downloading {BF16_NAME} (40.2 GB) to {directory}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, open(target, "ab" if partial else "wb") as out:
            if partial and response.status != 206:
                out.truncate(0)
                partial = 0
            done, last = partial, time.time()
            while chunk := response.read(16 << 20):
                out.write(chunk)
                done += len(chunk)
                if time.time() - last > 30:
                    print(f"  {done / GB:.1f} / {BF16_SIZE / GB:.1f} GB", flush=True)
                    last = time.time()
    print("Checking SHA-256 of the BF16 model...", flush=True)
    if sha256(target) != BF16_SHA256:
        raise RuntimeError(f"SHA-256 mismatch for {target}; delete it and download again")
    return target


def resolve(key_map, name, keys):
    target = key_map.get(name, name)
    fn = None
    if not isinstance(target, str):
        target, offset, fn = (*target, None)[:3]
        if offset is not None:
            raise NotImplementedError(f"{name}: adapters that replace part of a fused tensor are not supported")
    if target not in keys and target.endswith(".folded_bias"):
        target = target.replace(".folded_bias", ".linear.bias")
    if target not in keys:
        raise KeyError(f"{name} -> {target} is not in the template model")
    return target, fn


def tensor_bytes(tensor):
    return tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def merge(bf16_path, template_path, adapter_path, output_path, key_map, strength):
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    device = torch.device("cuda")
    template, bf16, adapter_file = TensorFile(template_path), TensorFile(bf16_path), TensorFile(adapter_path)
    header, metadata = template.header, dict(template.metadata)
    keys = set(header)
    adapter = {key: adapter_file.get(key) for key in adapter_file.header}
    adapter_file.close()
    bf16_header = {key: info["shape"] for key, info in bf16.header.items()}

    replaced = {}
    for key, value in adapter.items():
        if key.endswith((LORA_A, LORA_B)):
            continue
        target, fn = resolve(key_map, key, keys)
        value = fn(value) if fn is not None else value
        if list(value.shape) != header[target]["shape"]:
            raise ValueError(f"{key}: shape {list(value.shape)} != {header[target]['shape']}")
        replaced[target] = value
    loras = {}
    for root in sorted({key[:-len(LORA_A)] for key in adapter if key.endswith(LORA_A)}):
        target, fn = resolve(key_map, root + ".weight", keys)
        loras[target[:-len(".weight")]] = (adapter[root + LORA_A], adapter[root + LORA_B], fn)
    layers = sorted(key[:-len(".comfy_quant")] for key in keys if key.endswith(".comfy_quant"))
    if set(loras) - set(layers):
        raise ValueError("Adapter LoRA layers are not INT8 in the template: " + ", ".join(sorted(set(loras) - set(layers))[:5]))
    missing = [layer for layer in layers if bf16_header.get(layer + ".weight") != header[layer + ".weight"]["shape"]]
    if missing:
        raise ValueError("The BF16 model does not match the template, e.g. " + missing[0])
    print(f"INT8 layers {len(layers)}, LoRA layers {len(loras)}, replaced tensors {len(replaced)}", flush=True)

    metadata["pih3_fused"] = json.dumps({"bf16": bf16_path.name, "template": template_path.name, "adapter": adapter_path.name,
                                         "strength": strength, "merged": len(loras), "replaced": len(replaced), "quantized_once": True})
    # Template order, except that each INT8 weight follows its scale so only one quantized layer is held in RAM.
    # INT8 weights are multiples of 256 bytes, so the alignment of the following tensors is unchanged.
    order = []
    for key in sorted(header, key=lambda key: header[key]["data_offsets"][0]):
        if key.endswith(".weight") and key[:-len(".weight")] in layers:
            continue
        order.append(key)
        if key.endswith(".weight_scale") and key[:-len(".weight_scale")] in layers:
            order.append(key[:-len("_scale")])
    offset, new_header = 0, {"__metadata__": metadata}
    for key in order:
        start, end = header[key]["data_offsets"]
        new_header[key] = {"dtype": header[key]["dtype"], "shape": header[key]["shape"], "data_offsets": [offset, offset + end - start]}
        offset += end - start
    encoded = json.dumps(new_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    plain_same, started = [], time.time()
    with open(partial, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)) + encoded)
        # The INT8 weight and its scale are written separately; quantize a layer when the first of the two is reached.
        parts = {layer + suffix: (layer, suffix) for layer in layers for suffix in (".weight", ".weight_scale")}
        pending, finished = {}, 0
        for key in order:
            if key in parts:
                layer, suffix = parts[key]
                if layer not in pending:
                    quant = json.loads(bytes(template.get(layer + ".comfy_quant").tolist()).decode())
                    if quant != QUANT:
                        raise ValueError(f"{layer}: unexpected quantization {quant}")
                    weight = bf16.get(layer + ".weight").to(device, torch.float32)
                    if layer in loras:
                        a, b, fn = loras[layer]
                        delta = strength * (b.to(device, torch.float32) @ a.to(device, torch.float32))
                        weight += fn(delta) if fn is not None else delta
                    q, params = TensorWiseINT8Layout.quantize(weight, per_channel=True, convrot=True, convrot_groupsize=256)
                    if layer not in loras:
                        plain_same.append((q == template.get(layer + ".weight").to(device)).float().mean().item())
                    pending[layer] = {".weight": q.cpu(), ".weight_scale": params.scale.float().reshape(-1, 1).cpu()}
                    finished += 1
                    if finished % 25 == 0:
                        print(f"  {finished}/{len(layers)} layers ({time.time() - started:.0f} s)", flush=True)
                value = pending[layer].pop(suffix)
                if not pending[layer]:
                    del pending[layer]
            elif key in replaced:
                value = replaced[key].to(getattr(torch, {"BF16": "bfloat16", "F16": "float16", "F32": "float32"}[header[key]["dtype"]]))
            else:
                value = template.get(key)
            data = tensor_bytes(value)
            if len(data) != new_header[key]["data_offsets"][1] - new_header[key]["data_offsets"][0]:
                raise ValueError(f"{key}: unexpected byte size")
            out.write(data)
    template.close()
    bf16.close()
    os.replace(partial, output_path)
    print(f"Plain INT8 layers identical to the official quantization: mean {sum(plain_same) / len(plain_same):.6f}", flush=True)
    print(f"Saved {output_path} ({output_path.stat().st_size / GB:.1f} GB, {time.time() - started:.0f} s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comfy_argument", nargs="?", type=Path, help="ComfyUI folder passed by convert_fused.bat")
    parser.add_argument("python_argument", nargs="?", help="Python selected by convert_fused.bat")
    parser.add_argument("--comfy-root", type=Path, help="ComfyUI folder containing main.py")
    parser.add_argument("--bf16", type=Path, help=f"BF16 model ({BF16_NAME}); default: search models/diffusion_models")
    parser.add_argument("--download-bf16", type=Path, metavar="DIR", help="Download the BF16 model (40.2 GB) into DIR when --bf16 is not given")
    parser.add_argument("--delete-bf16", action="store_true", help="Delete the BF16 model after a successful conversion")
    parser.add_argument("--adapter", type=Path, help=f"pi-H3 adapter; default: {ADAPTER_NAME} in models/loras")
    parser.add_argument("--template", type=Path, help=f"official INT8 model; default: {TEMPLATE_NAME} in models/diffusion_models")
    parser.add_argument("--output", type=Path, help=f"default: models/diffusion_models/{OUTPUT_NAME}")
    parser.add_argument("--extra-model-paths", type=Path, help="additional ComfyUI extra_model_paths YAML")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    selected_root = args.comfy_root or args.comfy_argument or os.environ.get("H3_PIFLOW_SETUP_ROOT")
    if not selected_root:
        parser.error("Specify the ComfyUI folder (--comfy-root)")
    comfy_root = Path(selected_root).resolve()
    if not (comfy_root / "main.py").is_file():
        parser.error("--comfy-root must contain ComfyUI main.py")
    if sys.prefix == sys.base_prefix and Path(sys.executable).parent.name.lower() != "python_embeded":
        parser.error("Use the ComfyUI venv or portable Python; global/system Python is refused")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    key_map = load_piflow_key_map(comfy_root)
    folder_paths = model_paths(comfy_root, args.extra_model_paths)
    template = args.template or find_model(folder_paths, "diffusion_models", TEMPLATE_NAME)
    adapter = args.adapter or find_model(folder_paths, "loras", ADAPTER_NAME)
    output = args.output or Path(folder_paths.models_dir) / "diffusion_models" / OUTPUT_NAME
    if output.exists():
        parser.error(f"{output} already exists; delete it or choose another --output")
    if args.bf16:
        bf16 = args.bf16
    elif args.download_bf16:
        bf16 = download_bf16(args.download_bf16.resolve())
    else:
        bf16 = find_model(folder_paths, "diffusion_models", BF16_NAME)
    if shutil.disk_usage(output.parent if output.parent.exists() else comfy_root).free < template.stat().st_size + GB:
        raise RuntimeError(f"Not enough free space for the {template.stat().st_size / GB:.1f} GB output in {output.parent}")
    print(f"BF16     {bf16}\nTemplate {template}\nAdapter  {adapter}\nOutput   {output}", flush=True)
    merge(bf16.resolve(), template.resolve(), adapter.resolve(), output.resolve(), key_map, args.strength)
    if args.delete_bf16:
        bf16.unlink()
        print("Deleted the BF16 model:", bf16, flush=True)


if __name__ == "__main__":
    main()
