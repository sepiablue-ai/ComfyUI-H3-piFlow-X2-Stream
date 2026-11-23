# Attribution and model terms

Package code is GPL-3.0; see LICENSE. No model weights, converted weights or generated videos are distributed.

- [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) by Hansheng Chen (Apache-2.0) provides Load LakonLab Adapter and the pi-Flow Scheduler. The converter imports its MiniMax H3 key map (`lakonlab_loaders.minimax_h3_to_diffusers`) from the installed pack at runtime; no code is copied. The workflows follow its `pi-H3-Spark-SM89.json` settings, with the output changed to X2 Stream.
- The [pi-H3](https://huggingface.co/Lakonik/pi-H3) adapters by Lakonik identify their license as the MiniMax H3 Community License on the model page.
- [Spark-H3](https://github.com/zechengtang/Spark-H3) by zechengtang (Apache-2.0) provides MiniMaxH3SparkAttentionSM86, MiniMaxH3SparkAttentionSM89 and MiniMaxH3SparkAttentionSM120. Setup downloads the official release archive for the GPU generation (`ComfyUI-Spark-H3-0.2.0-windows-x86_64-{sm86,sm89,sm120}-cu130.zip`) and runs its `install.py`.
- [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream), by sepiablue-ai, is the separate GPL-3.0 X2 conversion, FFN chunking and NVENC dependency, fetched at commit e05e79ab3242abaafddd673e88e19c1824cd863c.
- The Comfy-Org MiniMax H3 weights, including the BF16 source and the INT8 ConvRot template used by the converter, identify their license as the MiniMax H3 Community License. A model built by the converter is a derivative of those weights and of the pi-H3 adapter and stays under their terms. The X2 VAE keeps the terms on its [distribution page](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE). This code license does not replace model or derived-weight terms.
- The converter uses the INT8 ConvRot quantization of comfy-kitchen, supplied by ComfyUI.
- examples/assets/fl2va_case01_no_text_948x1659.png is the same sample image distributed with ComfyUI-H3-X2-Stream.
