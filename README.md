# Ultraman LoRA — 让 SDXL 认识每一个奥特曼和怪兽

在单张 RTX 4090（24 GB）上，用自采数据把 SDXL base 1.0 微调成"报名字就出对应角色"的模型：
输入 `生迪迦`，出来的是迪迦，而不是随便一个奥特曼。

- **底模**：`stabilityai/stable-diffusion-xl-base-1.0`（fp16）
- **微调方式**：UNet-only LoRA，rank 32，1024 px，14000 步，单卡约 52 分钟
- **身份数**：34 个（24 个奥特曼 + 10 个怪兽），另有 12 位人间体/演员数据留档
- **中文提示**：内置 46 条中文名 → 英文 tag 归一表，中文直接可用

---

## 1. 它到底能做到什么

| 指标 | 底模 | 本 LoRA | 说明 |
|---|---|---|---|
| 身份识别准确率（34 身份 × N 张，DINOv2 最近质心 top-1） | <!--BASE_ACC--> | <!--LORA_ACC--> | 直接对应"生迪迦出来的是迪迦" |
| 通用能力漂移相似度（8 个非特摄提示，同种子 base vs LoRA） | 1.000 | <!--DRIFT--> | 越接近 1 越说明原能力没被破坏 |

验收脚本 `work/verify.py` 一次跑完这两道门 + 中文提示 demo，退出码 0 即 PASS。

## 2. 快速开始

```bash
# 1) 底模（约 7 GB）
python3 work/dl_sdxl.py

# 2) 生成
CUDA_VISIBLE_DEVICES=1 python3 work/infer.py \
    --lora ultraman-lora-r32.safetensors --prompt "生迪迦" --out diga.png

# 3) 一键验收
CUDA_VISIBLE_DEVICES=1 python3 work/verify.py --lora ultraman-lora-r32.safetensors --n 2
```

`--prompt` 支持中文名（见第 5 节对照表），也支持混写场景描述：
`生成赛罗奥特曼在东京街头战斗`。加 `--raw` 则原样送入 CLIP。

## 3. 数据是怎么来的

1. **角色名录**：先查清全量阵容再抓图，见 [`ROSTER.md`](ROSTER.md)
   （昭和/平成/新生代奥特曼、怪兽分级、39 位人间体与演员，含"向原龙"核实结论）。
2. **抓图**：Bing 图片搜索，真 Chromium（puppeteer-core）解析 `a.iusc` 的 `murl`，
   46 个查询共 7043 条 URL；下载器边下边存、64×64 缩略图 md5 去重、短边 ≥384、
   最长边 ≤1024、长宽比 ≤3:1，最终留存 5179 张。
3. **清洗**：DINOv2-base 提特征 → 每身份鲁棒质心（自相似前 60% 的均值）→
   保留 `own ≥ 0.55`（演员 0.38）且不输给"最像的别的身份"超过 0.10 的图 →
   每身份最多 32 张（演员 24）。奥特曼/怪兽 1370 张全部保留，其中 1088 张用于训练。
4. **演员数据**：12 位演员的图片单独留档（`data/raw/<id>`），**未参与训练**——
   搜索引擎对这些名字返回的大多是皮套剧照而非演员本人，质检不过关，硬训会污染模型。

## 4. 训练细节

- 只训练 UNet 的注意力投影（`to_q/to_k/to_v/to_out.0`），文本编码器冻结；
  先离线预计算 VAE latent 与双 CLIP 文本嵌入，训练时不再加载文本编码器。
- 噪声调度用 `EulerDiscreteScheduler.alphas_cumprod` 的原值（scaled_linear，
  β 0.00085→0.012），epsilon 预测，MSE 损失。
- bf16 + gradient checkpointing，batch 1，lr 1e-4（100 步 warmup + cosine），
  0.22 s/step，14000 步约 52 分钟，显存 14.7 GB。

### 两个必须知道的坑

1. **SDXL 的 VAE 在 fp16 下会溢出**：直接 `latent_dist.sample()` 出来的 latent 全是 NaN，
   loss 立刻变 NaN。必须把 VAE 转 fp32 再编码（本仓库已处理，并带 NaN 断言）。
2. **`CUDA_VISIBLE_DEVICES` 一定要设**：脚本里写 `cuda` 等于 GPU0。
   本机 GPU0 上有别人的训练和一个会杀掉占用 GPU0 的用户进程的 watchdog，
   不设这个变量，训练会在 15–35 秒内被 SIGTERM 掉，日志里只留下半行进度。

## 5. 中文名对照

`work/names.py` 里维护了 46 个身份的中文/英文别名（长名优先匹配，支持一句里多个角色）。
归一化示例：

```
生迪迦            -> tiga ultraman, giant silver and purple hero with a glowing chest crystal, ...
生成赛罗和泽塔打杰顿 -> ultraman zero, ..., ultraman z, ..., zetton, ...
一只猫            -> 一只猫   (不是特摄请求，原样交给底模)
```

覆盖：迪迦/赛罗/初代/赛文/杰克/艾斯/泰罗/雷欧/爱迪/戴拿/盖亚/阿古茹/高斯/奈克瑟斯/
梦比优斯/希卡利/银河/欧布/捷德/泽塔/特利迦/德凯/布莱泽/亚刻，以及哥莫拉/杰顿/巴尔坦/
雷德王/艾雷王/美菲拉斯/泰兰特/加坦杰厄/哥尔赞/安培拉。

## 6. 已知限制

- 训练图里混有海报/杂志页（带文字），模型偶尔会把文字画进图里。
- 设计相近的角色（初代 vs 梦比优斯、盖亚 vs 阿古茹）仍可能互串，这是数据本身的歧义。
- 只学了"角色长什么样"，没学剧情/必杀技/变身器，写复杂场景时构图能力来自底模。
- 演员/人间体未训练，只能生成"穿队服的泛化人物"。

## 7. 目录

```
work/            抓图、下载、清洗、训练、推理、评测、验收脚本
data/manifest.json     训练清单（图片路径 + 英文 caption + 身份）
data/clean_report.json 清洗报告（每身份原始数/保留数/平均相似度）
data/raw/        原始下载（按身份分目录，未入库）
samples/         验收生成的图
ROSTER.md        角色全量调研
```

## 8. 复现

```bash
python3 work/build_manifest.py --max-per-id 32 --sim 0.55 --sim-actor 0.38 \
        --max-actor 24 --min-actor 6 --min-per-id 10
CUDA_VISIBLE_DEVICES=1 python3 work/train_lora.py --manifest data/manifest.json \
        --steps 14000 --rank 32 --lr 1e-4 --save-every 1000 --kinds ultra,kaiju
CUDA_VISIBLE_DEVICES=1 python3 work/export_lora.py --ckpt out/lora_step14000.safetensors \
        --out ultraman-lora-r32.safetensors
```
