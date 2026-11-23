# ComfyUI H3 piFlow X2 Stream

[English](README.md) | 日本語

**RTX 4070 12GBで、音声付きの5秒動画（1088×1920）を32.3秒で生成できます。**

[pi-H3](https://huggingface.co/Lakonik/pi-H3)（pi-Flowで2stepに蒸留したMiniMax H3 FL2VA用アダプター）を、[Spark-H3](https://github.com/zechengtang/Spark-H3)のsparse attentionと[ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream)のINT8 X2 VAE・NVENC保存で組み合わせたワークフロー集です。544×960を2stepで生成し、X2 VAEで1088×1920に拡大して保存します。

| 構成（5秒・1088×1920・音声付き） | 生成時間 | 備考 |
|---|---:|---|
| 公式 pi-H3 DWD（1088×1920を直接生成） | 189.0秒（3分9.0秒） | ComfyUI-piFlow同梱の`pi-H3-DWD-Spark-SM89.json`。Peak RAM 17.7GB |
| **変換不要版**：動的LoRA + X2 Stream | **35.7秒** | 公式INT8モデルにアダプターを読み込むだけ |
| **最速版**：Fused + X2 Stream | **32.3秒** | 事前マージしたモデルを1回だけ作る。プロンプトを毎回変える実運用では約44〜45秒 |

<!-- TODO: paste GitHub video asset URL（3構成の比較動画） -->

数値はRTX 4070 12GB、ComfyUI 0.45.0での実測です（同じ画像・プロンプト・seed44、モデル読込済みの2回目、ComfyUIの`execution_start`から`execution_success`まで）。プロンプトを変えるたびにテキストエンコーダーの条件付け（約8.6秒）が入り、サンプリングも約2.5秒延びるため、別シーン5本では44.2〜45.1秒でした。

> 作者：かみもと（Kamimoto）｜検証結果の速報は **[X（@sep_is_heim）](https://x.com/sep_is_heim)**、解説記事は **[note](https://note.com/sepiablue)**（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）で公開しています。

## どちらのルートを使うか

| | 変換不要版（35.7秒） | 最速版 Fused（32.3秒） |
|---|---|---|
| ワークフロー | [piH3_2step_dynamic_x2_5s.json](workflows/piH3_2step_dynamic_x2_5s.json) | [piH3_2step_fused_x2_5s.json](workflows/piH3_2step_fused_x2_5s.json) |
| 拡散モデル | 公式 `minimax_h3_fl2va_pruned_int8_convrot`（21.0GB）＋ `pi_h3_2step_adapter`（503MB） | 変換で作る `minimax_h3_fl2va_pruned_pih3_2step_fused_int8_convrot`（21.0GB） |
| 準備 | モデルを置くだけ | `convert_fused.bat`で1回変換（約10分＋BF16のダウンロード） |
| 一時的に必要な空き容量 | なし | BF16元モデル40.2GB＋出力21.0GB。変換後にBF16は削除できます |
| 画質・音声 | 基準 | 耳で聴くと変換不要版と大差なし。構図・表情は同じseedでも少し変わります |

迷ったら変換不要版から始めてください。速さが欲しくなったら、同じ環境のまま最速版を追加できます（変換には公式INT8モデルとアダプターも使います）。

## サンプル動画

<!-- TODO: paste GitHub video asset URL（洋楽を歌う） -->

<!-- TODO: paste GitHub video asset URL（海辺を走る） -->

<!-- TODO: paste GitHub video asset URL（剣で斬り込む：高速アクションの崩れの例） -->

## 得意・不得意

最速版で別の5シーン（5秒・seed44）を作り、映像と音声を作者が評価しました。

| シーン | 映像 | 音声 |
|---|---|---|
| 洋楽を歌う（人物のアップ） | 5本で最も良い。顔・髪・肌が高精細で、カメラを指さす指先まで破綻なし | 英語の発音が自然で明瞭。リズム・口の動きも違和感なし |
| ギターソロ | ギター本体・表情は安定。フレットの指先と速いストロークの右手がやや溶ける | 速弾きの立ち上がり・フレーズが自然 |
| 海辺を走る | 走り方・水しぶき・表情が自然で破綻なし | セリフに軽い二重音声（位相ずれ感） |
| 剣で斬り込む | 速い振り抜きの瞬間に手先・剣がブロック状に崩れる。動きが止まるとくっきり戻る | BGM・掛け声・剣戟音は十分 |
| 邦楽を歌う | 動きがゆっくりで、着物・桜まで極めて鮮明 | 「あななたを待つ」のように音節が重複し、日本語のイントネーションが不自然 |

- **得意**：人物のアップ、ゆっくりした動き、英語の歌・セリフ、楽器演奏の音。
- **不得意**：剣の振り抜きのような高速アクション（2stepでは時間方向の補間が足りず、一時的にブロック状に崩れる）、日本語の歌・セリフ（音節のずれ、軽い二重音声）。
- **14秒以上は非推奨**：同じ2stepを14秒（345フレーム）で試すと、試聴で両seedとも音声がこもる・二重になる NG でした。5秒（124フレーム）での利用を想定しています。

## 必要な環境

WindowsとNVIDIA RTX 40シリーズ（SM89）向けです。開発検証はRTX 4070 12GB、ComfyUI 0.45.0、Python 3.13.13、PyTorch 2.14.0+cu130、comfy-kitchen 0.2.41で行いました。生成時のPeak VRAMは約11.6〜11.8GB、Peak RAMは約7.3〜8.4GBでした。実測はDynamic VRAM有効、`--disable-pinned-memory`で行いました。他のGPU・OS・モデルで同じ速度や品質を保証するものではありません。

Spark-H3にはPyTorchに合ったTriton（Windowsでは`triton-windows`。PyTorch 2.14では3.8.0.post28で検証）が必要です。入っていなければsetupは何も変更せずに止まります。

Spark-H3 0.2.0の公式対応はComfyUI 0.38〜0.40、0.42〜0.44です。このrepoの検証は0.45.0で行い、問題なく動きました。

このrepoにカスタムノードはありません。ワークフロー、変換スクリプト、導入用の`.bat`だけです。使うノードは次の3つのパックとComfyUI標準ノードです。

| パック | 用途 | setupが取得する版 |
|---|---|---|
| [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) | Load LakonLab Adapter、pi-Flow Scheduler | commit `e0ca46f7a7dcd080d8ab1d759719e8bc046617f0` |
| [Spark-H3](https://github.com/zechengtang/Spark-H3) | MiniMaxH3SparkAttentionSM89 | Release `comfyui-v0.2.0` のWindows SM89 cu130版（SHA-256確認） |
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | H3X2ChunkFeedForward、H3X2StreamSave、X2 VAEのINT8化 | commit `e05e79ab3242abaafddd673e88e19c1824cd863c` |

## 導入

### 1. 依存パックを入れる

ComfyUIを停止し、このrepoを任意のフォルダに置いて実行します。

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --install-dependencies
```

専用Pythonを省略すると、そのComfyUIのPortable Python、`.venv`、`venv`を探します。候補が1つでなければ明示指定が必要です。グローバルPythonは拒否します。setupは未導入のパックだけを上の版で取得し、Spark-H3は同梱の`install.py`でGPU用のwheelを入れます。既存のパックは更新しません。ComfyUI本体やPyTorchは更新せず、モデルもダウンロードしません。サンプル画像を`ComfyUI/input/`へコピーし、ノードの登録とモデルの配置を確認して、どちらのルートが使えるかを表示します。

ComfyUI-Managerで同じパックを入れている場合は、`--install-dependencies`を付けずに実行してください。フォルダ名が違っても、中身で見つけて二重に入れません。

検査のみ行う場合：

```bat
setup.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --check
```

### 2. モデルを置く

重み・動画は同梱しません。次のモデルを自分で配置してください。共有フォルダはComfyUI標準の`extra_model_paths.yaml`で指定できます（[設定例](extra_model_paths.yaml.example)）。

| ファイル | ComfyUI内の場所 | 使うルート | 配布元 |
|---|---|---|---|
| qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | models/text_encoders | 両方 | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/text_encoders) |
| minimax_h3_video_vae_int8_convrot.safetensors | models/vae | 両方 | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| minimax_h3_audio_vae_fp32.safetensors | models/vae | 両方 | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| MiniMax-H3-X2-Detail-v1.safetensors | models/vae | 両方 | [X2 Detail VAE](https://huggingface.co/speach1sdef178/MiniMax-H3-X2-Detail-VAE) |
| minimax_h3_fl2va_pruned_int8_convrot.safetensors | models/diffusion_models | 変換不要版（最速版の変換にも使用） | [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models) |
| pi_h3_2step_adapter.safetensors | models/loras | 変換不要版（最速版の変換にも使用） | [Lakonik/pi-H3](https://huggingface.co/Lakonik/pi-H3/tree/main/pi_h3_2step_adapter)の`diffusion_pytorch_model.safetensors`をこの名前で保存 |

X2 VAEのINT8版は、ComfyUI-H3-X2-Streamの`examples/00_prepare_int8.json`を一度実行して`models/vae/h3_x2_stream/`に準備します。

### 3. 最速版のモデルを作る（最速版を使う場合だけ）

ComfyUIを停止してから実行します。BF16元モデル（`minimax_h3_fl2va_pruned_bf16.safetensors`、40.2GB）を指定フォルダへダウンロードし、変換が成功したら削除します。

```bat
convert_fused.bat "C:\ComfyUI\ComfyUI" "C:\ComfyUI\.venv\Scripts\python.exe" --download-bf16 "D:\tmp\h3_bf16" --delete-bf16
```

- **空き容量**：BF16用に40.2GB（ダウンロード先）、出力用に21.0GB（`models/diffusion_models`）。変換後にBF16を消すので、最終的に増えるのは21.0GBです。
- **時間**：RTX 4070で変換は約10分です。ダウンロードは回線次第です（40.2GB）。
- **出力**：`models/diffusion_models/minimax_h3_fl2va_pruned_pih3_2step_fused_int8_convrot.safetensors`
- BF16をすでに持っている場合は`--bf16 PATH`で指定します（`--delete-bf16`を付けなければ残ります）。公式INT8モデルとアダプターは`models`から自動で探します。別の場所は`--template`、`--adapter`、`--output`で指定できます。
- ダウンロードは途中から再開でき、SHA-256を確認します。CUDAとComfyUIのPython（comfy-kitchen）が必要です。

#### 変換の中身

公式INT8 ConvRotモデルをひな形にして、INT8になっている200層だけをBF16から量子化し直します。アダプターが当たる150層は「BF16の重み＋LoRA差分」を、それ以外の50層はBF16の重みをそのまま、公式と同じ設定（行ごと、ConvRot、group 256）で1回だけ量子化します。アダプターのLoRA以外の57テンソル（adaln・最終層）はLoad LakonLab Adapterと同じく置き換えます。それ以外のテンソルは公式INT8モデルからそのままコピーします。

LoRAの当たらない50層は、公式INT8と99.9999%一致しました（公式と同じ量子化手順であることの確認）。公式INT8にLoRAを足してから量子化し直す方法（2回量子化）も試しましたが、LoRA差分がINT8の刻みより小さく、素の動的LoRAとの差が大きくなったため（PSNR 17.7〜19.3dB、BF16から1回だけの方式は20.7〜20.8dB）採用していません。

## ComfyUIで使う

1. [piH3_2step_dynamic_x2_5s.json](workflows/piH3_2step_dynamic_x2_5s.json)（変換不要版）か[piH3_2step_fused_x2_5s.json](workflows/piH3_2step_fused_x2_5s.json)（最速版）をComfyUIへドラッグします。API版は[workflows/api](workflows/api)です。
2. LoadImageで画像を選び、MiniMaxH3ImageToVideoのプロンプトを変更します。
3. 画像はImageScale（lanczos、crop=center）で544×960に合わせます。縦横比が違う画像は中央で切り抜かれます（例：1248×1824の画像は左右が約17%ずつ切れます）。解像度を変える場合は、ImageScaleとMiniMaxH3ImageToVideoのwidth・heightを同じ32の倍数にします。
4. 長さは`length`（5秒＝124フレーム）です。14秒以上は非推奨です（[得意・不得意](#得意不得意)）。

主な設定：SigmaShift 12/3、Spark-H3 speed（topk 20%、warmup 0）、pi-Flow Scheduler 2step（max_raw_t 0.93、final_step_size_scale 0.6）、euler、H3X2ChunkFeedForward chunks=0、24fps、NVENC CQ18。保存先はComfyUIのoutput/H3_piFlow_X2_Streamです。

## 検証範囲

2026年11月23日に、新規cloneしたComfyUI 0.45.0（commit dba377b9）と新規venv（Python 3.13.13、PyTorch 2.14.0+cu130、triton-windows 3.8.0.post28）で、別フォルダにcloneしたこのrepoから導入と生成を確認しました。モデルは既存のフォルダを`extra_model_paths.yaml`で参照しました。

- `setup.bat --install-dependencies`：3つのパックを取得し、Spark-H3のSM89 wheelを導入、全ノードの登録と変換不要版の準備完了を確認。Tritonがない環境では、何も変更せずに止まることも確認しました。
- `convert_fused.bat`：BF16のダウンロード（40.2GB、SHA-256一致）と変換が成功。変換は147秒、変換プロセスの最大メモリは2.7GBでした。出力の932テンソルはすべて、ベンチマークに使ったFusedモデルと完全に一致しました。
- 2本のワークフローをUI形式のままComfyUIに読み込んで実行し、どちらも成功。seed44・モデル読込済みで、最速版32.1秒、変換不要版36.9秒（初回はモデル読込を含め64.1秒）。同じ環境での再実行は出力が完全一致し、ベンチマーク環境の動画との差はPSNR 46.4dBでした。
- BF16の`--delete-bf16`による削除は、この試験では実行していません。X2 VAEのINT8版は準備済みのものを使い、`00_prepare_int8.json`は実行していません。ブラウザーでのドラッグ操作は、ワークフローを読み込む処理（`loadGraphData`）で代替しました。

## 作者・関連リポジトリ

かみもと（Kamimoto）です。RTX 4070 12GBで、MiniMax H3とComfyUIの高速化・省VRAM化を実測しながら公開しています。

- X：[@sep_is_heim](https://x.com/sep_is_heim) — step数・秒数・短縮率などの検証速報
- note：[かみもと｜ローカルAI生成](https://note.com/sepiablue) — 導入方法と実測の解説記事（[記事INDEX](https://note.com/sepiablue/n/n8e612c619a3e)）
- [さやかベンチマーク](https://sayakabenchmark.pages.dev/) — H3ワークフローの速度・画質比較

| リポジトリ | 内容 |
|---|---|
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | X2 VAEのINT8化と非同期NVENC保存。本repoの出力 |
| [ComfyUI-H3-AudioFirst-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-AudioFirst-X2-Stream) | 音声を先に作って固定する14秒向けワークフロー |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | 低解像度3step＋高解像度1stepのLBH 3+1 |

## ライセンス

本パッケージのコードはGPL-3.0です。モデル重み・変換した重みには各配布元の条件（MiniMax H3 Community License）が適用されます。詳細と出典は[NOTICE.md](NOTICE.md)を参照してください。
