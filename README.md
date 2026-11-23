# ComfyUI H3 piFlow X2 Stream

English | [日本語](README_ja.md)

**A 5 s video with audio at 1088x1920 in 32.3 s on an RTX 4070 12GB.**

Workflows that combine [pi-H3](https://huggingface.co/Lakonik/pi-H3) (a pi-Flow adapter that distills MiniMax H3 FL2VA to 2 steps) with the sparse attention of [Spark-H3](https://github.com/zechengtang/Spark-H3) and the INT8 X2 VAE and NVENC saving of [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream). The video is sampled at 544x960 in 2 steps and saved at 1088x1920 through the X2 VAE.

| Setup (5 s, 1088x1920, with audio) | Time | Notes |
|---|---:|---|
| Official pi-H3 DWD (sampled directly at 1088x1920) | 189.0 s (3 min 9.0 s) | `pi-H3-DWD-Spark-SM89.json` shipped with ComfyUI-piFlow. Peak RAM 17.7 GB |
| **No-conversion route**: dynamic LoRA + X2 Stream | **35.7 s** | Loads the adapter onto the official INT8 model |
| **Fastest route**: Fused + X2 Stream | **32.3 s** | Build a pre-merged model once. About 44-45 s in practice when the prompt changes every run |

Side by side, left to right: official DWD, no-conversion route, fastest route (each scaled to 544x960 with lanczos and placed side by side, libx264 CRF 18, no audio):

<!-- TODO: paste GitHub video asset URL (0_benchmark_side_by_side_dwd_dynamic_fused.mp4) -->

Official pi-H3 DWD (189.0 s):

https://github.com/user-attachments/assets/8bc43e9c-f53b-4fb7-8b63-e52604603109

No-conversion route, dynamic LoRA + X2 Stream (35.7 s):

<!-- TODO: paste GitHub video asset URL (2_benchmark_dynamic_lora_35.7s_x264.mp4) -->

Fastest route, Fused + X2 Stream (32.3 s):

<!-- TODO: paste GitHub video asset URL (3_benchmark_fused_32.3s_x264.mp4) -->

The three videos use the same image, prompt and seed 44. The first is the official workflow's SaveVideo output (libx264, CRF 18) as is. The other two were saved by X2 Stream with NVENC (CQ 18, about 11 MB), which exceeds GitHub's upload limit, so they were re-encoded for display with the same libx264 CRF 18 as the first. Resolution, frame count and audio are unchanged (the audio is copied), and the re-encoded video differs from the original by PSNR 45.5 dB / SSIM 0.993. The official DWD and the 2-step routes are different models, so the motion differs.

Measured on an RTX 4070 12GB with ComfyUI 0.45.0: same image, prompt and seed 44, second run with the models already loaded, from ComfyUI's `execution_start` to `execution_success`. A new prompt adds the text-encoder conditioning (about 8.6 s) and makes sampling about 2.5 s longer, so five other scenes took 44.2-45.1 s each.

> Author: Kamimoto | Quick results on **[X (@sep_is_heim)](https://x.com/sep_is_heim)**, articles on **[note](https://note.com/sepiablue)** ([article index](https://note.com/sepiablue/n/n8e612c619a3e), Japanese).

## Which route to use

| | No conversion (35.7 s) | Fastest, Fused (32.3 s) |
|---|---|---|
| Workflow | [piH3_2step_dynamic_x2_5s.json](workflows/piH3_2step_dynamic_x2_5s.json) | [piH3_2step_fused_x2_5s.json](workflows/piH3_2step_fused_x2_5s.json) |
| Diffusion model | Official `minimax_h3_fl2va_pruned_int8_convrot` (21.0 GB) + `pi_h3_2step_adapter` (503 MB) | `minimax_h3_fl2va_pruned_pih3_2step_fused_int8_convrot` (21.0 GB), built by the converter |
| Preparation | Place the models | Run `convert_fused.bat` once (about 2.5 min plus the BF16 download) |
| Temporary free space | None | 40.2 GB for the BF16 source + 21.0 GB output. The BF16 file can be deleted afterwards |
| Picture and audio | Reference | By ear about the same as the no-conversion route. Composition and expression change slightly at the same seed |

Start with the no-conversion route if unsure. The fastest route can be added to the same setup later (the converter also uses the official INT8 model and the adapter).

## Sample videos

Singing an English song (fastest route, 5 s, seed 44):

<!-- TODO: paste GitHub video asset URL (4_sample_english_song.mp4) -->

Running on a beach:

<!-- TODO: paste GitHub video asset URL (5_sample_beach_run.mp4) -->

Sword attack (example of block artifacts during fast motion):

<!-- TODO: paste GitHub video asset URL (6_sample_sword_attack.mp4) -->

## Strengths and weaknesses

Five other scenes (5 s, seed 44) were generated with the fastest route and reviewed by the author.

| Scene | Picture | Audio |
|---|---|---|
| Singing an English song (close-up) | Best of the five. Sharp face, hair and skin; the pointing fingers stay clean | Natural, clear English; rhythm and lip sync look right |
| Guitar solo | Guitar and face are stable. Fretting fingers and the fast strumming hand melt slightly | Natural attack and phrasing of the fast solo |
| Running on a beach | Running, splashes and expressions are natural | Slight doubled voice (phase-shifted feel) in the line |
| Sword attack | Hands and blades break into blocks during the fast swing and sharpen again when the motion stops | Music, shout and sword sounds work well |
| Singing a Japanese song | Slow motion; kimono and cherry blossoms are very sharp | Syllables repeat (e.g. "あななたを待つ") and the Japanese intonation is unnatural |

- **Good at**: close-ups of people, slow motion, English singing and speech, instrument sounds.
- **Weak at**: fast action such as a sword swing (2 steps do not interpolate enough in time, so details briefly break into blocks), Japanese singing and speech (syllable slips, slight doubled voice).
- **14 s or longer is not recommended**: the same 2-step setup at 14 s (345 frames) was rated NG by ear for both seeds (muffled and doubled audio). The workflows target 5 s (124 frames).

## Requirements

Windows with an NVIDIA RTX 40-series GPU (SM89). Developed and tested on an RTX 4070 12GB with ComfyUI 0.45.0, Python 3.13.13, PyTorch 2.14.0+cu130 and comfy-kitchen 0.2.41. Peak VRAM was about 11.6-11.8 GB and peak RAM about 7.3-8.4 GB while generating. Measured with Dynamic VRAM enabled and `--disable-pinned-memory`. Other GPUs, operating systems or models may give different speed and quality.

Spark-H3 needs the Triton build that matches PyTorch (`triton-windows` on Windows; tested with 3.8.0.post28 for PyTorch 2.14). Without it, setup stops before changing anything.

Spark-H3 0.2.0 officially supports ComfyUI 0.38-0.40 and 0.42-0.44. This repo was tested on 0.45.0 and worked.

This repo contains no custom nodes: only workflows, the converter and the setup batch files. The workflows use ComfyUI native nodes and these three packs.

| Pack | Used for | Version fetched by setup |
|---|---|---|
| [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) | Load LakonLab Adapter, pi-Flow Scheduler | commit `e0ca46f7a7dcd080d8ab1d759719e8bc046617f0` |
| [Spark-H3](https://github.com/zechengtang/Spark-H3) | MiniMaxH3SparkAttentionSM89 | Release `comfyui-v0.2.0`, Windows SM89 cu130 archive (SHA-256 checked) |
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | H3X2ChunkFeedForward, H3X2StreamSave, INT8 X2 VAE preparation | commit `e05e79ab3242abaafddd673e88e19c1824cd863c` |

## Installation

### 1. Install the packs

Stop ComfyUI, put this repo in any folder and run:

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --install-dependencies
```

Without the Python argument, setup looks for the Portable Python, `.venv` and `venv` of that ComfyUI and needs exactly one candidate. A global Python is refused. Setup fetches only missing packs at the versions above, and Spark-H3 installs its GPU wheel with its own `install.py`. Existing packs are not updated. ComfyUI itself and PyTorch are not changed and no model is downloaded. Setup copies the sample image to `ComfyUI/input/`, checks node registration and model placement, and reports which route is ready.

If the same packs were installed with ComfyUI-Manager, run setup without `--install-dependencies`. Packs are found by their contents, whatever the folder name, so nothing is installed twice.

Read-only check:

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --check
```

### 2. Place the models

No weights or videos are included. Place these models yourself. Shared folders can be set with ComfyUI's standard `extra_model_paths.yaml` ([example](extra_model_paths.yaml.example)).

| File | ComfyUI folder | Route | Source |
|---|---|---|---|
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | models/text_encoders | Both | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| minimax_h3_video_vae_int8_convrot.safetensors | models/vae | Both | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| minimax_h3_audio_vae_fp32.safetensors | models/vae | Both | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| MiniMax-H3-X2-Detail-v1.safetensors | models/vae | Both | [X2 Detail VAE](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) |
| minimax_h3_fl2va_pruned_int8_convrot.safetensors | models/diffusion_models | No conversion (also used by the converter) | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| pi_h3_2step_adapter.safetensors | models/loras | No conversion (also used by the converter) | `diffusion_pytorch_model.safetensors` from [Lakonik/pi-H3](https://huggingface.co/Lakonik/pi-H3/tree/main/pi_h3_2step_adapter), saved under this name |

Prepare the INT8 X2 VAE once with `examples/00_prepare_int8.json` of ComfyUI-H3-X2-Stream; it is written to `models/vae/h3_x2_stream/`.

### 3. Build the fastest-route model (fastest route only)

Stop ComfyUI first. This downloads the BF16 source model (`minimax_h3_fl2va_pruned_bf16.safetensors`, 40.2 GB) into the given folder and deletes it after a successful conversion.

```bat
convert_fused.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --download-bf16 "D:\tmp\h3_bf16" --delete-bf16
```

- **Free space**: 40.2 GB for the BF16 file (download folder) and 21.0 GB for the output (`models/diffusion_models`). Only 21.0 GB remains after the BF16 file is deleted.
- **Time**: 147 s of conversion on an RTX 4070 with an NVMe SSD, plus the 40.2 GB download and a SHA-256 check (about 2 min). The converter process stays at about 2.7 GB of memory.
- **Output**: `models/diffusion_models/minimax_h3_fl2va_pruned_pih3_2step_fused_int8_convrot.safetensors`
- If you already have the BF16 file, pass `--bf16 PATH` (it is kept unless `--delete-bf16` is given). The official INT8 model and the adapter are found in `models`; use `--template`, `--adapter` and `--output` for other locations.
- The download resumes after an interruption and is checked by SHA-256. CUDA and ComfyUI's Python (comfy-kitchen) are required.

#### What the converter does

The official INT8 ConvRot model is the template. Only its 200 INT8 layers are quantized again from BF16: the 150 adapter layers as BF16 weight + LoRA delta, the other 50 as the plain BF16 weight, once, with the official settings (per row, ConvRot, group 256). The adapter's 57 non-LoRA tensors (adaln and final layers) replace the template tensors, as Load LakonLab Adapter does. Every other tensor is copied from the official INT8 model.

The 50 layers without LoRA matched the official INT8 weights to 99.9999%, which confirms the same quantization procedure. Adding the LoRA to the official INT8 weights and quantizing again (double quantization) was also tested and rejected: the LoRA delta is smaller than one INT8 step, so the result drifted further from the dynamic LoRA (PSNR 17.7-19.3 dB against 20.7-20.8 dB for the single quantization from BF16).

## Using it in ComfyUI

1. Drag [piH3_2step_dynamic_x2_5s.json](workflows/piH3_2step_dynamic_x2_5s.json) (no conversion) or [piH3_2step_fused_x2_5s.json](workflows/piH3_2step_fused_x2_5s.json) (fastest) into ComfyUI. API versions are in [workflows/api](workflows/api).
2. Choose an image in LoadImage and edit the prompt in MiniMaxH3ImageToVideo.
3. ImageScale (lanczos, crop=center) fits the image to 544x960. Images with another aspect ratio are center-cropped (a 1248x1824 image loses about 17% of its width in total). To change the resolution, set the same multiples of 32 in ImageScale and MiniMaxH3ImageToVideo.
4. `length` sets the duration (5 s = 124 frames). 14 s or longer is not recommended ([Strengths and weaknesses](#strengths-and-weaknesses)).

Main settings: SigmaShift 12/3, Spark-H3 speed (top-k 20%, no warmup), pi-Flow Scheduler 2 steps (max_raw_t 0.93, final_step_size_scale 0.6), euler, H3X2ChunkFeedForward chunks=0, 24 fps, NVENC CQ18. Output goes to ComfyUI's output/H3_piFlow_X2_Stream.

## Validation

On 2026-11-23, installation and generation were checked from a separate clone of this repo on a freshly cloned ComfyUI 0.45.0 (commit dba377b9) with a new venv (Python 3.13.13, PyTorch 2.14.0+cu130, triton-windows 3.8.0.post28). Models were read from an existing folder through `extra_model_paths.yaml`.

- `setup.bat --install-dependencies` fetched the three packs, installed the Spark-H3 SM89 wheel, and reported all nodes registered and the no-conversion route ready. Without Triton it stopped before changing anything.
- `convert_fused.bat` downloaded the BF16 model (40.2 GB, SHA-256 matched) and converted it in 147 s with a peak of 2.7 GB of process memory. All 932 output tensors were identical to the fused model used for the benchmarks.
- Both workflows were loaded in UI format and ran successfully. Seed 44 with models loaded: fastest route 32.1 s, no-conversion route 36.9 s (64.1 s on the first run, including model loading). Reruns in the same environment gave identical outputs; the difference from the benchmark environment's videos was PSNR 46.4 dB.
- Deleting the BF16 file with `--delete-bf16` was not exercised in this test. The INT8 X2 VAE was already prepared, so `00_prepare_int8.json` was not run. Dragging in a browser was replaced by the workflow loading call (`loadGraphData`).

## Author and related repositories

Kamimoto. I measure and publish speed and VRAM improvements for MiniMax H3 and ComfyUI on an RTX 4070 12GB.

- X: [@sep_is_heim](https://x.com/sep_is_heim) - quick results such as steps, seconds and speed-ups
- note: [Kamimoto | Local AI generation](https://note.com/sepiablue) - installation guides and measurement articles ([article index](https://note.com/sepiablue/n/n8e612c619a3e), Japanese)
- [Sayaka Benchmark](https://sayakabenchmark.pages.dev/) - speed and quality comparison of H3 workflows

| Repository | Contents |
|---|---|
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | INT8 X2 VAE and asynchronous NVENC saving; the output stage of this repo |
| [ComfyUI-H3-AudioFirst-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-AudioFirst-X2-Stream) | Generates and locks the audio first; 14 s workflows |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | LBH 3+1: 3 low-resolution steps + 1 high-resolution step |

## License

The package code is GPL-3.0. Model weights and converted weights keep the terms of their sources (MiniMax H3 Community License). See [NOTICE.md](NOTICE.md) for details and attribution.
