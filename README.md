# Ego Action Privacy

第一视角视频动作隐私保护基线：3D/逐帧 U-Net 生成有界 RGB 扰动，冻结的
InternVideo2-CLIP-1B-224p-f8 提供输入梯度，只更新扰动生成器。

## 当前状态（2026-10-15）

- 默认协议已切换为官方 Ego-Exo4D Fine-grained Keystep Recognition v2：278 类、
  ego-only，直接沿用官方 baseline 仓库的 `label_id`、train/val 样本与顺序。
- 官方 train 14,326 条、val 4,517 条已按 `(take_name,start_time,end_time)` 与本地
  原始标注逐条精确匹配：0 缺失、0 歧义，所有视频均可用；训练覆盖 278 类，验证
  覆盖 272 类。
- 已缓存 `[278,768]` 文本特征。完整 4,517 条官方验证集的 278-way zero-shot
  clean 基线为 **Top-1 7.239%（327/4,517）**、Top-5 24.972%（1,128/4,517）、
  真类平均排名 30.51。
- 原 665 类完整词表协议及结果没有删除，保留在 `configs/full_vocab_665.yaml` 作为
  长尾/开放词表扩展实验；其 Top-1 5.113% 不属于官方 benchmark。
- 官方下载器当前没有实际下发文档所列的 `label_mapping.csv`；本项目用官方 baseline
  仓库的 CSV `label_id` 与发布的原始 `step_unique_id/name` 做一致性连接来重建映射，
  并在报告中保存来源 URL 与 SHA256。
- 像素来自本地官方 `downscaled/448` ego 完整视频，并在官方 CSV 的原始起止时间上
  采样；没有另行使用再次编码的 `keystep_benchmark/clips_448p` 预裁剪 MP4。类别、
  划分和区间是官方协议，编码层面可能与预裁剪文件存在细微差异。
- 原始 665 类范围仍为 train 15,839 条、val 4,871 条。清单保留原始区间和异常行，
  可用于审计与扩展协议。
- 项目接手时有视频的样本是 train 3,354、val 1,138。随后从官方 448p manifest
  精确补齐 514 个已解析 RGB 文件和 4 个唯一 Aria RGB 路径，不下载 preview、SLAM
  或 exo 流。最终 train 有效且有视频 15,624 条/600 个实际可用类别，val 有效且有
  视频 4,870 条/447 类。
- 仍有 18 个训练 take（213 条有效片段）既不在最新官方 metadata，也不在官方
  448p manifest；另有 3 条原始标注时间区间无效。它们全部保留在审计清单，正式
  全有效训练默认保持阻塞。
- 下载后 650 个可用训练视频逐文件真实解码 0 失败；官方验证集 4,517 个动作区间
  在新 baseline 中逐条解码和推理成功。
- 已严格组合 Stage2 视觉基础权重、InternVL 文本基础权重和 CLIP 适配权重，缓存
  文本特征。真实视频前向与 RGB 输入梯度检查通过，目标
  参数保持冻结。
- 真实视频 5-step smoke 已确认生成器参数更新、攻击 margin 总体下降且扰动满足
  `epsilon=8/255`。这些 smoke 数字不构成正式隐私效果结论。

默认 278 类官方协议的 train/val 均已完整可用。数据加载仍使用 `valid` 策略：任何
有效标注缺视频都会拒绝启动，不会把局部数据静默称作“全量”。原 665 类协议仍因
18 个官方发布中缺失的训练 take 而保持正式全量训练阻塞。

## 方法

输入和输出统一为 `[B,T,3,H,W]`、RGB `[0,1]`：

```text
clean video -> U-Net -> epsilon*tanh(raw perturbation)
            -> clip(clean + delta, 0, 1)
            -> frozen InternVideo2-CLIP -> 278 class-text logits
            -> untargeted margin + magnitude + temporal + optional spatial-TV
```

`generator.temporal_mode: 3d` 使用只在空间降采样的 3D U-Net；改为
`framewise` 会对每帧独立应用共享 2D U-Net，是时间模块消融。时序损失权重可设为
零，形成另一项独立消融。同预算随机噪声 `U[-epsilon,epsilon]` 已集成到评估脚本。

## 环境与目录

服务器已有可用环境实际位于：

```bash
PY=/data2/yhao/projects/ego_privacy/py310/bin/python
export PYTHONPATH=$PWD/src
```

无需重装整套 CUDA/PyTorch。已实查 Python 3.10.12、PyTorch 2.9.0+cu128、
torchvision 0.24.0+cu128、CUDA 12.8 和 RTX PRO 6000。`pyproject.toml` 记录了依赖；
源码直接通过 `PYTHONPATH=src` 运行，不修改旧环境。

默认官方 278 类路径在 [configs/default.yaml](configs/default.yaml) 中；原 665 类协议
保存在 [configs/full_vocab_665.yaml](configs/full_vocab_665.yaml)。原始数据与模型
只读复用，生成物写到 `/data2/yhao/outputs/ego_action_privacy`。

## 数据准备与审计

生成完整清单（存在缺失时返回码 2 是预期的审计阻塞，不代表清单没有写出）：

```bash
$PY scripts/build_manifests.py \
  --data-root /data2/yhao/datasets/egoexo4d \
  --output-dir /data2/yhao/outputs/ego_action_privacy/manifests
```

从官方 baseline 仓库的 v2 ego CSV 生成 278 类清单；连接必须逐条精确成功：

```bash
$PY scripts/build_official_benchmark.py \
  --train-csv /path/to/official/fine_grained/data/files_v2/ego/train.csv \
  --val-csv /path/to/official/fine_grained/data/files_v2/ego/val.csv \
  --raw-train-manifest /data2/yhao/outputs/ego_action_privacy/manifests/keystep_train.json \
  --raw-val-manifest /data2/yhao/outputs/ego_action_privacy/manifests/keystep_val.json \
  --output-dir /data2/yhao/outputs/ego_action_privacy/official_278/manifests
```

逐视频或逐片段解码：

```bash
$PY scripts/audit_decode.py \
  --manifest /data2/yhao/outputs/ego_action_privacy/manifests/keystep_val.json \
  --output /data2/yhao/outputs/ego_action_privacy/manifests/val_decode_audit.json
```

官方 CLI 的 `--views ego` 对 448p manifest 中 `views=None` 的路径不能排除 exo/SLAM
流。以下脚本严格按 `video_audit.json` 已选择的单个 RGB 相对路径规划，默认不下载；
确认报告后才加 `--yes`：

```bash
/data2/yhao/envs/egoexo/bin/python scripts/download_missing_rgb.py \
  --data-root /data2/yhao/datasets/egoexo4d \
  --video-audit /data2/yhao/outputs/ego_action_privacy/manifests/video_audit.json \
  --report /data2/yhao/outputs/ego_action_privacy/manifests/rgb_download_plan.json
```

下载后必须重新生成清单和解码审计；脚本不会替 18 个缺元数据或 4 个流歧义的
take 猜测相机。

若官方 448p manifest 对某个未解析 take 恰好只有一个明确的 Aria `214-1` RGB
路径，可显式加 `--include-unique-official-aria-fallbacks --overrides /path/overrides.json`。
重新构建时再传 `--video-overrides /path/overrides.json`。override 会记录官方 manifest、
选择理由和 SHA256；不满足“唯一 Aria RGB”的 take 仍保持阻塞。

## 目标模型检查与文本缓存

278 类是原 665 类词表的有序子集，可直接复用已验证的相同文本嵌入：

```bash
$PY scripts/subset_text_cache.py \
  --source-class-map /data2/yhao/outputs/ego_action_privacy/manifests/class_map.json \
  --source-cache /data2/yhao/outputs/ego_action_privacy/cache/keystep_665_text_features.pt \
  --target-class-map /data2/yhao/outputs/ego_action_privacy/official_278/manifests/class_map.json \
  --output /data2/yhao/outputs/ego_action_privacy/official_278/cache/keystep_278_text_features.pt

$PY scripts/smoke_target.py \
  --model-dir /data2/yhao/models/InternVideo2-CLIP-1B \
  --manifest /data2/yhao/outputs/ego_action_privacy/official_278/manifests/keystep_train.json \
  --class-map /data2/yhao/outputs/ego_action_privacy/official_278/manifests/class_map.json \
  --text-cache /data2/yhao/outputs/ego_action_privacy/official_278/cache/keystep_278_text_features.pt \
  --report /data2/yhao/outputs/ego_action_privacy/smoke/target_gradient.json
```

适配器内部执行 224×224 bicubic resize 和 ImageNet 归一化；扰动在归一化前的
`[0,1]` 像素域添加。文本只添加一次 `summarize:` 前缀，最大 80 token。固定文本
特征缓存后，训练只加载视觉塔，释放 7B 文本塔的显存。
生成器的工作分辨率由 `data.image_size` 控制，首版为 224；数据层只做 RGB resize，
不做 ImageNet 归一化，因而不会与目标 wrapper 重复归一化。

## 训练与评估

官方 278 类数据上的正式训练（默认拒绝 partial data）：

```bash
$PY scripts/train.py --config configs/default.yaml \
  --run-dir /data2/yhao/outputs/ego_action_privacy/official_278/runs/unet3d_eps8
```

中断后可追加 `--resume /path/to/checkpoint.pt`；恢复时会严格检查生成器结构配置，
并恢复 AdamW 状态和全局 step。

仅做程序 smoke 时用显式样本与步数限制：

```bash
$PY scripts/train.py --config configs/default.yaml \
  --run-dir /data2/yhao/outputs/ego_action_privacy/smoke/train \
  --limit 2 --max-steps 2
```

clean baseline：

```bash
$PY scripts/evaluate.py --config configs/default.yaml \
  --output-dir /data2/yhao/outputs/ego_action_privacy/official_278/baseline/clean_valid_val
```

保护视频、clean 和同预算随机噪声三方比较：

```bash
$PY scripts/evaluate.py --config configs/default.yaml \
  --checkpoint /path/to/checkpoint_final.pt \
  --output-dir /data2/yhao/outputs/ego_action_privacy/official_278/eval/unet3d_eps8
```

输出包括 Top-1/Top-5、真类平均排名、`Linf`、MAE、PSNR、SSIM、扰动帧间差，
以及训练未见类别子集。`examples/*.mp4` 从左到右是 clean、protected、放大映射后的
delta。

消融实验不需要改代码：训练命令追加 `--temporal-mode framewise` 得到逐帧 U-Net，
追加 `--temporal-weight 0` 关闭时序损失；二者分别运行，避免同时改变两个因素。

## 测试

```bash
PYTHONPATH=src $PY -m pytest -q
```

测试覆盖原始与官方类别映射、官方行的精确连接、显式 partial-data 策略、采样边界、
U-Net 两种模式、预算、损失、冻结目标梯度和文本缓存与 class map 的绑定。
InternVideo2 派生代码的许可与修改说明见 `LICENSE-InternVideo` 和
`NOTICE-InternVideo`。
