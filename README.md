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
| 身份识别准确率，8 个**指标已验证**的身份（DINOv2 最近质心 top-1） | 18.8% | **18.8%** | 真照片在同一个指标上 88.3%；生成图 vs 真人剧照有域差，这个数不能用来判断对错，见下文 |
| 通用能力漂移相似度（8 个非特摄提示，同种子 base vs LoRA，`lora-scale 0.9`） | 1.000 | **0.775** | 越接近 1 越说明原能力没被破坏；PASS 线 0.75，8 张全部无退化。`scale 1.0` 实测 0.758 也 PASS，想加强身份可以调到 1.0，代价是猫/雪山/跑车这几类画面变化更大（0.445 是雪山那一张） |
| 34 身份全表准确率（仅供参考） | 7.4% | 12% | **这个指标本身不成立**：喂真照片也只有 49.8% |
| 名字影响比（换名字的像素差 / 换种子的像素差） | 0.463 | **0.520** | 必须比底模高；绝对值 <1 说明构图噪声仍大于名字的影响 |

为什么全表数字不能当结论：把每个身份的照片对半切开做留出验证，**真照片**在 34 个候选里
只有 49.8% 能认回自己（迪迦 44%、德凯 0%）。DINOv2-base 的全局特征分不开这一族
"银红配色 + 发光眼 + 胸口计时器 + 头冠"的相似设计，候选集一大就互相覆盖。
所以本项目只在真照片自召回 ≥70% 的 8 个身份上报数字，其余 26 个奥特曼靠人工看图验收。
细节见 [`docs/evaluation.md`](docs/evaluation.md) 第 8、9 节。

交付的 fp16 权重已实测与验收所用 fp32 权重等价（同种子渲染平均像素差 0.13%，
sha256 与远端导出件一致），溯源见 [`docs/evaluation.md`](docs/evaluation.md) 第 11 节。

验收脚本 `work/verify.py` 一次跑完四道门（原能力回归、名字影响比、身份准确率仅参考、渲染非空），
退出码 0 即 PASS。**身份正确性必须人眼看**：`compare_sheet.jpg` 是「参考照片 / base / LoRA-纯名字 /
LoRA-名字+描述」四行同种子对照表，三轮验收的分数与逐条缺陷记在
[`docs/acceptance-rounds.md`](docs/acceptance-rounds.md)。

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
3. **第一道清洗（语义）**：DINOv2-base 提特征 → 每身份鲁棒质心（自相似前 60% 的均值）→
   保留 `own ≥ 0.55`（演员 0.38）且不输给"最像的别的身份"超过 0.10 的图 →
   每身份最多 32 张（演员 24）。奥特曼/怪兽 1370 张通过。
4. **第二道清洗（结构）**：语义过滤抓不到两类脏图，因为它们和角色本身很像——
   **双人拼图**（左右两格都是同一角色，相似度当然高）和**电视截图/杂志扫描**
   （带黑边、日文字幕、水印）。`work/filter_hard.py` 用两个不依赖语义的信号剔掉它们：
   贯穿 ≥45% 画面的硬接缝（拼图边界/装订线）、四边 5.5% 边带标准差 < 7（黑边/白页边）。
   1370 张里再剔掉 703 张，剩 **667 张、每身份 12–21 张**（`data/clean2/`）。
   这一步是必须的：只做语义过滤时，模型会忠实学会"拼图 + 字幕 + 水印"的画风。
5. **演员数据**：12 位演员的图片单独留档（`data/raw/<id>`），**未参与训练**——
   搜索引擎对这些名字返回的大多是皮套剧照而非演员本人，质检不过关，硬训会污染模型。

## 4. 训练细节

- 只训练 UNet 的注意力投影（`to_q/to_k/to_v/to_out.0`），文本编码器冻结；
  先离线预计算 VAE latent 与双 CLIP 文本嵌入，训练时不再加载文本编码器。
- 噪声调度用 `EulerDiscreteScheduler.alphas_cumprod` 的原值（scaled_linear，
  β 0.00085→0.012），epsilon 预测，MSE 损失。
- bf16 + gradient checkpointing，batch 1，100 步 warmup + cosine，0.25 s/step，显存 14.7 GB。
- **四轮训练，最后一轮才是交付物**（每一步都是被验收打回来的）：
  - `out_v2/`：14000 步、lr 1e-4、只用语义清洗的 1370 张。角色能画对，但画风被脏数据带偏
    ——输出里出现双人拼图、日文字幕、杂志水印。
  - `out_c2/`：5000 步、lr 5e-5、结构清洗 667 张。**欠拟合**：迪迦出成胸像特写、
    哥莫拉画错。教训是砍学习率可以，砍步数不行。
  - `out_c3/`：12000 步、lr 1e-4、667 张。验收 47.55/100：纯名字已能绑上 4/6 的角色特征，
    但**配色错**（哥莫拉画成银灰、迪迦胸口画成蓝紫晶体）、构图漂移、文字污染。
  - `out_c4/`：12000 步、lr 1e-4、结构清洗 869 张 + **caption 全部重写**。
    第 1 轮验收定位到的根因是 caption 里没写颜色（`gomora: ancient horned dinosaur
    monster` 一个颜色词都没有），所以 29 条 caption 全部改成"配色 + 头部特征 + 胸口特征"。
- **推理端默认带负面提示词**（`names.NEGATIVE`）：`text, watermark, subtitle, logo,
  multiple people, collage, split screen, close-up portrait, cropped head, deformed`。
  训练图是抓来的网页图，天然带字幕水印拼图，负面提示词是压制这些的最便宜手段。
- **caption dropout 0.4**：40% 的样本用"只有名字"的短 caption（`tiga ultraman, tokusatsu,
  full body`），60% 用完整描述。没有这一步，模型只学会"描述→长相"，而不会把名字
  当成触发器——裸名字提示就失效。

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

## 6. 三轮验收的结果（如实）

对照表 `compare_sheet.jpg` 经 3 轮独立评判，综合分 **47.55 → 51.15 → 57.55**，未达 85 门槛。
逐列最终判定（`lora@0.9 desc`）：

| 列 | 判定 |
|---|---|
| 迪迦 | **PASS**（银白 + 红紫 + 金色圆形计时器 + 头冠） |
| 赛罗 | FAIL（双人物同框） |
| 哥莫拉 | PASS（弱，灰身恐龙 + 独角 + 尾；缺红色背棘） |
| 杰顿 | **PASS** |
| 初代 | PASS（角色）/ FAIL（画面有 "ULTRAMAN" 水印、半身） |
| 盖亚 | PASS（弱，缺金色胸口 V） |

**瓶颈是数据不是参数**：每身份只有 12–28 张网页抓图，带水印/拼图；哥莫拉的灰体色是
训练图本身就是低饱和灰调（棕色像素占比 0.21、灰色 0.44）导致的，改 caption 无效。
下一轮的正确做法是**补图**（每身份 40 张以上、优先官方设定图），而不是调步数或学习率。
逐条缺陷与根因见 [`docs/acceptance-rounds.md`](docs/acceptance-rounds.md)。

## 7. 已知限制

- **身份准确率没有可靠的自动数字。** 三个候选指标全部失效（候选集太大、生成图 vs 真人
  剧照的域差、姿态背景主导全局特征），细节见 [`docs/evaluation.md`](docs/evaluation.md)
  第 8–10 节。身份对不对只能靠看图，仓库里的 `compare_sheet.jpg` 就是给人看的对照表。
- **训练图只有 12–28 张/身份**（688 张 / 34 个身份）。这是本项目最大的瓶颈：
  角色 LoRA 通常一个概念就要 20–50 张，34 个概念共用 rank 32 是偏紧的。
  要显著提升，第一步是每身份补到 40 张以上，而不是调训练参数。
- 设计相近的角色（初代 vs 梦比优斯、盖亚 vs 阿古茹、迪迦的三种形态）仍可能互串。
  迪迦在设定里有复合/强力/空中三种形态，训练图混着三种配色，模型会学成一个平均值。
- 怪兽的图源比奥特曼差（多为剧照截图），哥莫拉这类只有 12 张可用，还原度弱于奥特曼。
- 只学了"角色长什么样"，没学剧情/必杀技/变身器，写复杂场景时构图能力来自底模。
- 演员/人间体未训练，只能生成"穿队服的泛化人物"——搜索演员名返回的多是皮套剧照。

## 8. 目录

```
work/            抓图、下载、清洗、训练、推理、评测、验收脚本
data/manifest.json     训练清单（图片路径 + 英文 caption + 身份）
data/clean_report.json 清洗报告（每身份原始数/保留数/平均相似度）
data/raw/        原始下载（按身份分目录，未入库）
samples/         验收生成的图
ROSTER.md        角色全量调研
```

## 9. 复现

```bash
python3 work/build_manifest.py --max-per-id 32 --sim 0.55 --sim-actor 0.38 \
        --max-actor 24 --min-actor 6 --min-per-id 10
CUDA_VISIBLE_DEVICES=1 python3 work/train_lora.py --manifest data/manifest.json \
        --steps 14000 --rank 32 --lr 1e-4 --save-every 1000 --kinds ultra,kaiju
CUDA_VISIBLE_DEVICES=1 python3 work/export_lora.py --ckpt out/lora_step14000.safetensors \
        --out ultraman-lora-r32.safetensors
```
