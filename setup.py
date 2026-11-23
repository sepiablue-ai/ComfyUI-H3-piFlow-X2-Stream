"""Use an existing ComfyUI Python; no ComfyUI launch, model download or core edits."""
import argparse
import asyncio
import hashlib
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
    ("ComfyUI-Spark-H3", "comfyui_spark_split.py",
     "https://github.com/zechengtang/Spark-H3/releases/download/comfyui-v0.2.0/ComfyUI-Spark-H3-0.2.0-windows-x86_64-sm89-cu130.zip",
     "ComfyUI-Spark-H3/", "d610d19859fffe0eda69f0075749c4b4c1f5cd0ff0bc4967cbfac3466e007391"),
)
NODES = ("MiniMaxH3ImageToVideo", "MiniMaxH3SigmaShift", "SamplerCustomAdvanced", "BasicGuider", "VAEDecodeAudio",
         "Load LakonLab Adapter", "pi-Flow Scheduler", "MiniMaxH3SparkAttentionSM89", "H3X2ChunkFeedForward", "H3X2StreamSave")
COMMON_MODELS = {
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": ("minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_audio_vae_fp32.safetensors"),
}
X2_INT8 = "MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors"
X2_SOURCE = "MiniMax-H3-X2-Detail-v1.safetensors"
DYNAMIC_MODELS = {"diffusion_models": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "loras": "pi_h3_2step_adapter.safetensors"}
FUSED_MODEL = "minimax_h3_fl2va_pruned_pih3_2step_fused_int8_convrot.safetensors"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def find_package(comfy_root, marker):
    return next((path for path in sorted((comfy_root / "custom_nodes").iterdir()) if (path / marker).is_file()), None)


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
    matches = [item for item in folder_paths.get_filename_list(folder) if Path(item).name == name]
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
                        help="Install missing X2 Stream, ComfyUI-piFlow and Spark-H3 (Windows SM89) into this dedicated Python")
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

    packages = {}
    for name, marker, url, prefix, digest in DEPENDENCIES:
        found = find_package(comfy_root, marker)
        if found is None and args.install_dependencies:
            print(f"Installing {name} ...", flush=True)
            found = download_package(comfy_root, name, url, prefix, digest)
            if name == "ComfyUI-H3-X2-Stream":
                pip_install("--no-deps", "-r", str(found / "requirements.txt"))
            if name == "ComfyUI-Spark-H3":
                subprocess.run([sys.executable, str(found / "install.py")], check=True)
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
    missing_nodes = [name for name in NODES if name not in nodes.NODE_CLASS_MAPPINGS]
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
                      "x2_int8_decoder": x2 or "not prepared yet: run examples/00_prepare_int8.json of ComfyUI-H3-X2-Stream once",
                      "routes_ready": routes, "missing_common_models": missing, "models": models,
                      "video_generations": 0, "model_downloads": 0, "core_files_changed": False}, ensure_ascii=True, indent=2))
    if not ready:
        sys.exit("Place the missing models (see README.md) and run setup.bat --check again")


if __name__ == "__main__":
    main()
