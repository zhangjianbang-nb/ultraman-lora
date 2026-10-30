# 验收记录（≤3 轮，按 AGENTS.md 最大验收轮次铁律）

验收口径：**人工/子代理看图**。三个自动身份指标全部失效（见 `evaluation.md` 第 8–10 节），
唯一可信的自动指标是原能力回归（0.792，PASS）。

对照表 `compare_sheet.jpg`：4 行 × 6 列。
第 1 行 reference photo（该身份最接近质心的真实照片）、第 2 行 base（未微调）、
第 3 行 LoRA 纯名字、第 4 行 LoRA 名字+外观描述。同列同种子。

---

## 第 1 轮（模型 c3：clean2 667 张 / 12000 步 / lr 1e-4 / dropout 0.5）

综合分 **47.55 / 100 — FAIL**

| 维度 | 权重 | 分 |
|---|---|---|
| ①视觉与角色还原 | 0.35 | 48 |
| ②画面干净度 | 0.25 | 35 |
| ③功能正确性（名字是否控制画面） | 0.25 | 58 |
| ④风险与可回退性 | 0.15 | 50 |

逐列（以 `lora@1.0 desc` 为准）：tiga FAIL / zero FAIL / gomora FAIL / **zetton PASS** /
ultraman FAIL / gaia PASS（边缘）。

**关键结论**：纯名字路径**已经绑上 4/6 的身份特征**（赛罗头镖、哥莫拉独角、杰顿黑橙、
盖亚头冠）——绑定机制是通的。失分集中在三类可修的缺陷：

1. **配色错**：哥莫拉被画成银灰金属色（应褐色）；盖亚蓝色占比过高（应红银蓝）；
   迪迦胸口是蓝紫菱形晶体（应是金色计时器）。根因是 caption 里**没写颜色**：
   `gomora: ancient horned dinosaur monster with a nasal horn` —— 一个颜色词都没有。
2. **构图错**：迪迦出成半身特写（prompt 里有 "full body" 也没用）。根因是迪迦的
   训练图里特写占比高。
3. **脏数据泄漏**：初代格里有红色日文字「ゴモラ」、赛罗格是双人物并排。根因是训练图
   本身带文字/拼图，而且**对照表第 1 行的参考照片自己就是脏拼图**（迪迦参考图带中文标题
   和第二个黑影人物、哥莫拉参考图带"哥莫拉/红王"水印）——这说明结构过滤还没干净。

## 第 1 轮 → 第 2 轮的修复动作

- **caption 全部重写**（29 条）：每个身份都写明配色 + 头部特征 + 胸口特征。
  例：`gomora: brown and grey horned dinosaur monster with a nasal horn and a thick tail`、
  `tiga: silver body with red and purple accents, a golden chest timer and a crest fin`。
  caption 同时用于训练和 desc 推理路径，两边一起改。
- **数据换 clean3**：黑边判据从"边带标准差 < 7"改成"又平（std<3）又黑或又白"，
  避免误杀纯色影棚背景的好图；1370 → 869 张（训练用 688 张，每身份 12–28 张）。
- **参考照片改用 clean3**：对照表第 1 行换成结构过滤后的图，避免拿脏拼图当标准答案。
- 训练：c4 = clean3 + 新 caption，12000 步 / lr 1e-4 / rank 32 / dropout 0.5。
