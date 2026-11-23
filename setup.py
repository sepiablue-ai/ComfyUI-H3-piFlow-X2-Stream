"""Use an existing ComfyUI Python; no ComfyUI launch, model download or core edits."""
import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
# (folder name, marker file, archive URL, archive root, SHA-256 of the archive or None for a pinned commit)
DEPENDENCIES = (
    ("ComfyUI-H3-X2-Stream", "streaming.py",
     "https://codeload.github.com/sepiablue-ai/ComfyUI-H3-X2-Stream/zip/e05e79ab3242abaafddd673e88e19c1824cd863c",
     "ComfyUI-H3-X2-Stream-e05e79ab3242abaafddd673e88e19c1824cd863c/", None),
    ("ComfyUI-piFlow", "lakonlab_loaders.py",
     "https://codeload.github.com/Lakonik/ComfyUI-piFlow/zip/e0ca46f7a7dcd080d8ab1d759719e8bc046617f0",
     "ComfyUI-piFlow-e0ca46f7a7dcd080d8ab1d759719e8bc046617f0/", None),
    ("ComfyUI-Spark-H3", "comfyui_spark_split.py", None, "ComfyUI-Spark-H3/", None),  # URL and SHA-256 from SPARK_ARCHIVES
)
SPARK_RELEASE = "https://github.com/zechengtang/Spark-H3/releases/download/comfyui-v0.2.0/"
# Official Windows cu130 archives of Spark-H3 comfyui-v0.2.0, one per GPU generation
SPARK_ARCHIVES = {
    "sm86": ((8, 6), "ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm86-cu130.zip",
             "83e2455162ff7e9550c6fdcbda5534c7768dbecf60593cbb3223deeb7fe6c5d1"),
    "sm89": ((8, 9), "ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm89-cu130.zip",
             "d610d19859fffe0eda69f0075749c4b4c1f5cd0ff0bc4967cbfac3466e007391"),
    "sm120": ((12, 0), "ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm120-cu130.zip",
              "87034b19523bd02090bb54f7fae1f872ff5edbff9b60ec8ae1de93938ef8ca23"),
}
NODES = ("MiniMaxH3ImageToVideo", "MiniMaxH3SigmaShift", "SamplerCustomAdvanced", "BasicGuider", "VAEDecodeAudio",
         "Load LakonLab Adapter", "pi-Flow Scheduler", "H3X2ChunkFeedForward", "H3X2StreamSave")
COMMON_MODELS = {
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": ("minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_audio_vae_fp32.safetensors"),
}
X2_INT8 = os.path.join("h3_x2_stream", "MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors")
X2_SOURCE = "MiniMax-H3-X2-Detail-v1.safetensors"
DYNAMIC_MODELS = {"diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "loras": "pi_h3_2step_adapter.safetensors"}
FUSED_MODEL = "minimax_h3_fl2va_pruned_pih3_2step_fused_int8_convrot.safetensors"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def find_package(comfy_root, marker):
    return next((path for path in sorted((comfy_root / "custom_nodes").iterdir()) if (path / marker).is_file()), None)


def visible_gpus():
    """(CUDA_VISIBLE_DEVICES entry, name, architecture or None, SMxx) for each GPU this Python can see."""
    import torch
    if not torch.cuda.is_available():
        return []
    entries = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    by_capability = {capability: arch for arch, (capability, _, _) in SPARK_ARCHIVES.items()}
    gpus = []
    for index in range(torch.cuda.device_count()):
        capability = tuple(torch.cuda.get_device_capability(index))
        gpus.append((entries[index] if index < len(entries) else str(index), torch.cuda.get_device_name(index),
                     by_capability.get(capability), "SM%d%d" % capability))
    return gpus


def installed_architecture(package):
    build = package / "spark_h3_build.json" if package else None
    if build is None or not build.is_file():
        return None
    return json.loads(build.read_text(encoding="utf-8")).get("architecture")


def select_architecture(requested, installed, gpus):
    if installed and requested and installed != requested:
        sys.exit(f"Spark-H3 for {installed.upper()} is already installed in this ComfyUI; --arch {requested} does not "
                 "match. One ComfyUI can hold only one Spark-H3 build. Nothing was changed.")
    if requested or installed:
        return requested or installed
    found = sorted({gpu[2] for gpu in gpus if gpu[2]})
    if not found:
        seen = ", ".join(f"{gpu[1]} ({gpu[3]})" for gpu in gpus) or "no CUDA GPU"
        sys.exit(f"No supported GPU found (SM86 / SM89 / SM120); this Python sees {seen}. Nothing was changed.")
    if len(found) > 1:
        sys.exit("GPUs of different generations are visible (" + ", ".join(a.upper() for a in found) +
                 "); choose one with --arch sm86|sm89|sm120. Nothing was changed.")
    return found[0]


def download_package(comfy_root, name, url, prefix, digest):
    destination = comfy_root / "custom_nodes" / name
    request = urllib.request.Request(url, headers={"User-Agent": "H3-piFlow-X2-Setup"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if digest and hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("SHA-256 mismatch for " + url)
    archive = zipfile.ZipFile(io.BytesIO(data))
    members = []
    for member in archive.infolist():
        if not member.filename.startswith(prefix):
            raise RuntimeError("Unexpected archive root in " + url)
        relative = Path(member.filename[len(prefix):])
        if relative.is_absolute() or ".." in relative.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise RuntimeError("Unsafe archive path in " + url)
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise RuntimeError("Archive escapes its directory: " + url)
        members.append((member, target))
    destination.mkdir(parents=True, exist_ok=False)
    for member, target in members:
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    return destination


def pip_install(*arguments):
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode == 0:
        command = [sys.executable, "-m", "pip", "install"]
    elif shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable]
    else:
        raise RuntimeError("Neither pip nor uv is available for this Python")
    subprocess.run(command + list(arguments), check=True)


def find_model(folder_paths, folder, name):
    candidates = folder_paths.get_filename_list(folder)
    matches = [name] if name in candidates else [item for item in candidates if Path(item).name == name]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple copies of {name} in models/{folder}; keep one")
    return folder_paths.get_full_path_or_raise(folder, matches[0]) if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_argument", nargs="?", type=Path, help="ComfyUI folder passed by setup.bat")
    parser.add_argument("python_argument", nargs="?", help="Python selected by setup.bat")
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--check", action="store_true", help="Read-only environment and model check")
    parser.add_argument("--install-dependencies", action="store_true",
                        help="Install missing X2 Stream, ComfyUI-piFlow and Spark-H3 (Windows) into this dedicated Python")
    parser.add_argument("--arch", choices=tuple(SPARK_ARCHIVES),
                        help="Spark-H3 build: sm86 (RTX 30), sm89 (RTX 40) or sm120 (RTX 50). Detected from the GPU "
                             "when omitted; required when GPUs of different generations are visible")
    parser.add_argument("--extra-model-paths", type=Path, help="Additional native ComfyUI model-path YAML")
    args = parser.parse_args()
    if args.check and args.install_dependencies:
        parser.error("--check cannot be combined with installation")
    selected_root = args.comfy_root or args.comfy_argument or os.environ.get("H3_PIFLOW_SETUP_ROOT")
    if not selected_root:
        parser.error("Specify the ComfyUI folder")
    comfy_root = Path(selected_root).resolve()
    if not (comfy_root / "main.py").is_file() or not (comfy_root / "comfy/sd.py").is_file():
        parser.error("--comfy-root must contain ComfyUI main.py and comfy/sd.py")
    portable = Path(sys.executable).resolve().parent.name.lower() == "python_embeded"
    if sys.prefix == sys.base_prefix and not portable:
        parser.error("Use the existing ComfyUI venv or portable Python; global/system Python is refused")

    spark_package = find_package(comfy_root, "comfyui_spark_split.py")
    if args.install_dependencies and spark_package is None and importlib.util.find_spec("triton") is None:
        sys.exit("Spark-H3 needs the Triton build that matches this PyTorch (triton-windows on Windows; "
                 "PyTorch 2.14 was tested with triton-windows 3.8.0.post28). Install it into this Python first; nothing was changed.")
    gpus = visible_gpus()
    architecture = select_architecture(args.arch, installed_architecture(spark_package), gpus)
    matching = [gpu for gpu in gpus if gpu[2] == architecture]
    if args.install_dependencies and spark_package is None and not matching:
        sys.exit(f"Installing Spark-H3 {architecture.upper()} needs a visible {architecture.upper()} GPU; "
                 "none was found. Nothing was changed.")
    _, spark_archive, spark_digest = SPARK_ARCHIVES[architecture]
    print(f"Spark-H3 architecture: {architecture.upper()}" + (f" ({matching[0][1]})" if matching else
          " (no matching GPU visible; the Spark node will refuse to run on this PC)"), flush=True)
    packages = {}
    for name, marker, url, prefix, digest in DEPENDENCIES:
        if name == "ComfyUI-Spark-H3":
            url, digest = SPARK_RELEASE + spark_archive, spark_digest
        found = find_package(comfy_root, marker)
        if found is None and args.install_dependencies:
            print(f"Installing {name} ...", flush=True)
            found = download_package(comfy_root, name, url, prefix, digest)
            if name == "ComfyUI-H3-X2-Stream":
                pip_install("--no-deps", "-r", str(found / "requirements.txt"))
            if name == "ComfyUI-Spark-H3":
                # install.py checks the current CUDA device, so let it see only a GPU of the selected generation
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=matching[0][0])
                if subprocess.run([sys.executable, str(found / "install.py")], env=environment).returncode != 0:
                    shutil.rmtree(found)
                    sys.exit("Spark-H3 install.py failed (see above); its folder was removed so setup can be run again")
        if found is None:
            raise RuntimeError(f"{name} is missing; install it or use --install-dependencies")
        packages[name] = str(found)

    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(comfy_root))
    sys.argv = [sys.argv[0], "--disable-pinned-memory"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths
    from utils.extra_config import load_extra_path_config
    config = comfy_root / "extra_model_paths.yaml"
    if config.is_file():
        load_extra_path_config(str(config))
    if args.extra_model_paths:
        load_extra_path_config(str(args.extra_model_paths.resolve()))
    import nodes
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    spark_node = "MiniMaxH3SparkAttention" + architecture.upper()
    missing_nodes = [name for name in NODES + (spark_node,) if name not in nodes.NODE_CLASS_MAPPINGS]
    if missing_nodes:
        raise RuntimeError("Nodes not registered (check the ComfyUI log above): " + ", ".join(missing_nodes))

    models, missing = {}, []
    for folder, names in COMMON_MODELS.items():
        for name in (names,) if isinstance(names, str) else names:
            models[name] = find_model(folder_paths, folder, name)
            if models[name] is None:
                missing.append(f"models/{folder}/{name}")
    x2 = find_model(folder_paths, "vae", X2_INT8)
    if x2 is None and find_model(folder_paths, "vae", X2_SOURCE) is None:
        missing.append(f"models/vae/{X2_SOURCE}")
    routes = {}
    for folder, name in DYNAMIC_MODELS.items():
        models[name] = find_model(folder_paths, folder, name)
    routes["dynamic"] = all(models[name] for name in DYNAMIC_MODELS.values())
    models[FUSED_MODEL] = find_model(folder_paths, "diffusion_models", FUSED_MODEL)
    routes["fused"] = models[FUSED_MODEL] is not None
    if not args.check:
        for source in (ROOT / "examples/assets").glob("*"):
            target = Path(folder_paths.get_input_directory()) / source.name
            if target.exists() and sha(target) != sha(source):
                raise RuntimeError("Existing sample image differs; preserve it: " + str(target))
            if not target.exists():
                shutil.copy2(source, target)
    ready = not missing and (routes["dynamic"] or routes["fused"])
    print(json.dumps({"status": ("check_passed" if args.check else "installed") if ready else "models_missing",
                      "python": sys.executable, "comfy_root": str(comfy_root), "packages": packages,
                      "spark_architecture": architecture, "spark_node": spark_node,
                      "workflows": str(ROOT / "workflows" / architecture),
                      "x2_int8_decoder": x2 or "not prepared yet: run examples/00_prepare_int8.json of ComfyUI-H3-X2-Stream once",
                      "routes_ready": routes, "missing_common_models": missing, "models": models,
                      "video_generations": 0, "model_downloads": 0, "core_files_changed": False}, ensure_ascii=True, indent=2))
    if not ready:
        sys.exit("Place the missing models (see README.md) and run setup.bat --check again")


if __name__ == "__main__":
    main()
