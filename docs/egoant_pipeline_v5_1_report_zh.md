# EgoANT 标注管线 V5.1 中文报告

日期：2026-07-16

这份报告总结了我们基于 WGO-Bench 对 EgoANT 标注管线做的一轮实验。重点记录：
尝试过哪些想法、各自得分如何、三个评分指标怎么定义，以及目前最强的可复现流程。

## 样例文件

仓库中包含以下本地样例（源视频不随仓库分发）：

- [全帧 contact sheet](examples/pipeline_report_assets/homer1_contact_sheet_84_94.jpg)。
- [手部/物体 crop sheet](examples/pipeline_report_assets/homer1_handcrop_sheet_84_94.jpg)。
- [单视频 gold/pred 对比](examples/wgo_homer_1_compare/README.md)。

这里的 crop sheet 是从 `homer_1` 同一段时间窗口里裁出来的手部/物体视图。后面固定边界 hand-crop 消融已经重新跑过：我们先用 HaWoR 生成 `hand_recon.json` 的 wrist track，再基于 hand track 做 crop。因此新分数是真 hand-reconstruction crop，不是之前的 center-proxy crop。

## 官方参考

参考来源：

- Macrodata WGO-Bench blog:
  https://macrodata.co/blog/annotating-robot-video-subtasks
- WGO-Bench dataset:
  https://huggingface.co/datasets/macrodata/WGO-Bench
- Scale dense video captioning blog:
  https://labs.scale.com/blog/path-to-large-scale-dense-video-captioning

Macrodata 报告里的官方参考数值：

| 任务 | 官方参考 |
| --- | ---: |
| 分段 | `0.306` Segment F1 |
| 固定边界标注 | `61.0%` accuracy |
| 端到端 | `0.168` Semantic E2E F1 |

WGO-Bench 总共有 100 个 episode、743 个 gold subtask segment。HomER 子集有 25 个第一视角 episode、470 个 gold segment。

## 指标定义

### 1. Segment F1

Segment F1 只评估时间边界，不看文字标注。

1. 如果 predicted segment 和 gold segment 的 temporal IoU 大于等于 `0.75`，就认为时间上 match。
2. 匹配是 greedy one-to-one，一个 predicted segment 不能同时匹配多个 gold segment。
3. 我们使用 WGO-style outer-boundary snap：每个 episode 的第一个 predicted start 和最后一个 predicted end 会被 snap 到 episode 边界。
4. 所有 episode 上做 micro precision、recall、F1：

```text
precision = temporal_matches / predicted_segments
recall    = temporal_matches / gold_segments
F1        = 2 * precision * recall / (precision + recall)
```

这个指标只回答一个问题：模型有没有找到正确的 subtask 时间段。

### 2. 固定边界 Annotation Accuracy

这个指标固定 gold 边界，只评估文字标注质量。

1. 给模型一个 gold segment 的起止时间。
2. 模型给这个 segment 生成 subtask label。
3. judge model 判断 predicted label 和 gold label 是否语义等价。
4. Accuracy：

```text
accuracy = semantically_correct_labels / gold_segments
```

我们的本地消融主要使用 Qwen3.5-397B judge。Macrodata 官方数值用的是他们自己的评测栈，所以 judge calibration 不完全 apples-to-apples。

### 3. Semantic E2E F1

端到端指标同时评估时间分段和语义标注。

1. 先按 Segment F1 的规则匹配 predicted/gold 时间段。
2. 只有时间上 match 的 pair 才送进 semantic judge。
3. 计算 semantic precision、recall、F1：

```text
semantic_precision = semantic_matches / predicted_segments
semantic_recall    = semantic_matches / gold_segments
semantic_E2E_F1    = 2 * semantic_precision * semantic_recall
                     / (semantic_precision + semantic_recall)
```

所以 E2E F1 会远低于固定边界 annotation accuracy。多切、漏切、时间没对上，都会先伤害分母；文字标注再好也救不回来。

## 实验记录

### A. Raw Frames vs Overlay

原始 EgoANT 生产标注主要用 raw segment frames，加上 hand/geometry context。我们测试过视觉 overlay 是否能帮助 VLM，结果没有帮助。

| 固定边界标注输入 | 模型 | Accuracy |
| --- | ---: | ---: |
| raw frames | Qwen3.5-397B | `50.2%` |
| light overlay | Qwen3.5-397B | `48.5%` |
| raw frames | Qwen3.6-27B | `55.7%` |
| overlay | Qwen3.6-27B | `50.6%` |

结论：raw visual input 仍然是最稳的默认输入。Overlay 没有带来收益。

### B. Temporal Collage / Contact Sheets 用于标注

我们测试过把 timestamped temporal collage 或相邻 segment sheet 给 labeler，而不是只给当前 segment 的 raw frames。

| 固定边界标注输入 | 模型 | Accuracy |
| --- | ---: | ---: |
| raw frames | Qwen3.5-397B | `50.2%` |
| temporal collage | Qwen3.5-397B | `45.1%` |
| neighbor contact sheet | Qwen3.5-397B | `36.8%` |
| neighbor contact sheet + timestamp rerun | Qwen3.5-397B | `35.5%` |
| temporal collage | Qwen3.6-27B | `52.8%` |
| neighbor contact sheet | Qwen3.6-27B | `34.0%` |

结论：contact sheet 对分段有帮助，但单独拿来做固定边界 subtask labeling 不够好。它容易给模型更多上下文，但也会让模型混淆当前 segment 到底是哪一步。

### C. Hand Crops / Hand Collage

最开始的 hand-focused crop/collage 实验有一个重要 caveat：WGO-Bench HomER 本身没有 `hand_recon.json` 或现成 hand asset。所以旧的 WGO hand-collage 路径其实用的是 center-proxy crop / debug hand-collage approximation，不是真正的 EgoANT production `raw-hand-crop`。

| 固定边界标注输入 | 模型 | Accuracy |
| --- | ---: | ---: |
| raw frames | Qwen3.5-397B | `50.2%` |
| hand collage, center proxy | Qwen3.5-397B | `40.6%` |
| true hand-recon crop + raw frames | Qwen3.5-397B | `51.1%` |
| raw frames | Qwen3.6-27B | `55.7%` |
| hand collage, center proxy | Qwen3.6-27B | not re-scored |

production EgoANT 的 `raw-hand-crop` 路径不一样：如果有 `joints_xy` 就用 joints；没有就 fallback 到 `wrist_xy`；然后围绕可见手部点裁剪，并把 2x3 crop grid 追加到 raw frames 后面。

后续复跑结果：我们为全部 25 个 HomER episode 生成 HaWoR hand reconstruction 后，true hand-crop 路径得分是 `239/470 = 50.85%`，略高于 Qwen3.5-397B raw-frame baseline 的 `236/470 = 50.21%`。输入质量整体可用：`411/470` 个 segment 有 `5/5` 张有效 crop，其余是 partial hand-track coverage。

结论：旧 hand-collage 分数低，主要说明 center-proxy crop 不行，不能证明真正 hand crop 不行。真 hand-reconstruction crop 有一点正向收益，尤其对局部手部动作动词有帮助；但提升很小，而且要先跑昂贵的 hand-reconstruction prepass。所以当前建议是：

- 没有 hand asset 时，raw frames 仍然作为默认。
- 已经有可靠 hand asset，或者生成 hand asset 成本低时，可以把 true hand crop 作为辅助 view。
- 目前不建议为了 `+0.4pp` 的 HomER fixed-boundary gain 强行给所有数据跑 hand recon。

### D. Contact-Sheet Segmentation

在时间分段任务上，contact sheet 的帮助明显更大。当前 HomER 最强分段 recipe 是 timestamped 20-frame sheets：每 0.5 秒抽 1 帧，每张 sheet 20 帧，约覆盖 10 秒。

HomER-only segmentation 消融：

| 分段条件 | 模型 | Segment F1 | pred/gold | matches |
| --- | ---: | ---: | ---: | ---: |
| EgoANT 原始线 | pipeline | `0.0953` | 810/470 | 61 |
| Contact sheet legacy max3 | Qwen3.5-397B | `0.0952` | 265/470 | 35 |
| Contact sheet legacy max3 | Qwen3.6-27B | `0.1278` | 281/470 | 48 |
| Whole episode + legacy | Qwen3.6-27B | `0.1230` | 148/470 | 38 |
| Whole episode + GEPA aligned | Qwen3.6-27B | `0.1369` | 202/470 | 46 |
| S1 anti-underseg | Qwen3.5-397B | `0.1556` | 558/470 | 80 |
| S2 refine full25 | Qwen3.5-397B | `0.1674` | 498/470 | 81 |
| S2 pad=0.0 + full-cover prompt | Qwen3.6-27B | `0.2031` | 308/470 | 79 |

结论：Qwen3.6-27B + S2 full-cover prompt 是目前 HomER 上最强分段器。它的主要收益来自减少过度切分，同时还能保住足够多的时间匹配。

### E. 相邻 Segment Merge

我们测试过把相邻 predicted segments 后处理合并。视觉上更干净，但 Segment F1 下降。

| Merge mode | pred | matches | Segment F1 |
| --- | ---: | ---: | ---: |
| none | 308 | 79 | `0.2031` |
| exact same label | 305 | 77 | `0.1987` |
| verb/object | 290 | 74 | `0.1947` |
| short bridge | 263 | 69 | `0.1883` |

结论：不要盲目 merge。有些看起来丑的碎片，在 benchmark temporal matching 上反而有用。

### F. Relabeling And Candidate Selection

当 S2 full-cover 预测出边界后，label quality 就成为下一瓶颈。我们在同一批 HomER predicted boundaries 上测试了多个 relabeling 路径。

| HomER E2E variant | pred/gold | IoU match | semantic match | Segment F1 | E2E F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EgoANT one-pass baseline | 810/470 | 61 | 42 | `0.0953` | `0.0656` |
| 27B S2 full-cover, self labels | 308/470 | 79 | 41 | `0.2031` | `0.1054` |
| + Qwen3.5-397B raw relabel | 308/470 | 79 | 54 | `0.2031` | `0.1388` |
| + Qwen3.5-397B ffmpeg raw relabel | 308/470 | 79 | 55 | `0.2031` | `0.1414` |
| + 27B-prior neighbor relabel | 308/470 | 79 | 42 | `0.2031` | `0.1080` |
| + raw397-prior neighbor relabel | 308/470 | 79 | 56 | `0.2031` | `0.1440` |
| + Qwen3.5-397B candidate selector | 308/470 | 79 | 60 | `0.2031` | `0.1542` |

结论：当前 HomER E2E 最强是 hybrid：

- Qwen3.6-27B 负责分段。
- Qwen3.5-397B 负责标注和 candidate selection。

WGO-style seeded relabeling 的思路有部分迁移价值，但不能照搬。对我们的 Qwen stack 来说，397B raw label 比 27B segmentation label 更适合作为默认 prior。

### G. Universal vs Family-Specific Full WGO

我们也测试了 HomER-best recipe 是否可以直接用到所有 WGO subset。结论是不行。

| Pipeline | Full-WGO E2E F1 | Notes |
| --- | ---: | --- |
| GEPA coarse 397B self-label | `0.1048` | 之前完整跑过的 full-WGO run |
| S2 refine 397B pad=0.0 self-label | `0.0699` | robot subsets 上过度切分 |
| Universal HomER candidate-selector pipeline | `0.1098` | HomER/Galaxea/RoboInter 都用同一个 recipe |
| Family-specific mix | `0.1624` | HomER candidate-selector + robot GEPA coarse |
| Official Macrodata reference | `0.1680` | 公开 full-WGO reference |

Universal HomER pipeline 的 subset 细节：

| Subset | pred/gold | IoU match | semantic match | Segment F1 | E2E F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HomER | 308/470 | 79 | 59 | `0.2031` | `0.1517` |
| Galaxea | 163/123 | 25 | 12 | `0.1748` | `0.0839` |
| RoboInter | 280/150 | 26 | 11 | `0.1209` | `0.0512` |
| Full WGO | 751/743 | 130 | 82 | `0.1740` | `0.1098` |

结论：需要 family-specific policy。HomER 这种 human egocentric 视频适合 S2 full-cover；robot 视频目前更适合 GEPA coarse segmentation。这个 full-WGO transfer 表是历史结果，不能和后续 Gemini-3.5-Flash HomER-only 复评结果混合成同一个最终分数。

## 当前最强流程

当前 HomER 上测得的最强流程是：

```text
video
-> 对完整 episode 生成 timestamped contact sheets
   - 每 0.5 秒抽 1 帧
   - 每张 sheet 20 帧
   - 每张 sheet 约覆盖 10 秒
-> Qwen3.6-27B S2 full-cover temporal segmentation
-> 得到 predicted segment boundaries + draft labels
-> 为每个 predicted segment 生成多个候选 label
   A. Qwen3.5-397B raw-frame label
   B. Qwen3.5-397B raw-prior neighbor relabel
   C. Qwen3.5-397B 27B-prior neighbor relabel
   D. Qwen3.5-397B ffmpeg raw-frame label
-> Qwen3.5-397B candidate selector
   - 默认偏向 A，即 raw label
   - 只有其他候选明显更符合视频时才替换
   - neighbor context 只作为纠错证据，不作为主标签来源
-> final predicted boundaries + selected subtask labels
```

对 WGO 之外的真实 production data，建议把两条路线分开：

1. 生产安全默认路线：
   - 使用已有 fixed-boundary 或 wrist-minima segmentation。
   - 使用 raw-frame Qwen3.5-397B subtask labeling。
   - 如果已有可靠 hand asset，可以额外加入 true hand-recon crop view。
   - 保留 conservative judge/rewrite/quality gate。
2. 实验性 model segmentation 路线：
   - WGO-style timestamped contact sheets。
   - Qwen3.6 S2 full-cover segmentation。
   - Qwen3.5-397B candidate-selector labeling。

这个区分很重要。WGO model segmentation 很有希望，但它目前仍然是 benchmark-driven 的实验路线；对于已经有可靠边界的数据集，不应该静默替换掉稳定生产管线。

## 当前理解

1. 分段仍然是主要 ceiling。HomER 上，即使当前 predicted boundaries 的 label 全部正确，E2E 也会被 Segment F1 `0.2031` 限住。
2. Qwen3.6-27B 在 HomER temporal segmentation 上明显强于 Qwen3.5-397B。
3. Qwen3.5-397B 仍然是更好的 fixed-boundary labeler。
4. Raw frames 是最强的 no-asset labeling visual input。Overlay、temporal collage、neighbor sheets、旧 WGO center-proxy hand-collage 都弱于 raw frames。
5. True hand-reconstruction crops 在 HomER fixed-boundary labeling 上略强于 raw frames：`50.9%` vs `50.2%`。但收益小，且需要 hand-recon 成本。
6. Candidate selection 有用，因为 raw label 错的时候，正确标签经常已经出现在某个候选里。
7. Full WGO 必须做 family-specific routing。Universal HomER-tuned pipeline 会伤害 robot subsets。

## 下一步建议

1. 改进 candidate selector：更严格限制什么时候可以 override raw label。
2. 把 true hand crop 作为候选之一加入 selector，而不是直接替换 raw frames。比如 A=raw，B=neighbor relabel，C=ffmpeg raw，D=true hand-crop。
3. 只在 full-frame 里手/物体区域太小，或者已有可靠 hand asset 时使用 hand crop。
4. 加 adaptive segmentation routing：
   - HomER-like egocentric human videos：S2 full-cover。
   - Robot videos：GEPA coarse。
5. 如果需要严格 apples-to-apples，对 best family-specific full-WGO output 用官方/Gemini judge 再复评一次。

## 给项目决策的一句话总结

目前最值得保留的核心改进是：用 Qwen3.6 做 contact-sheet 分段，用 Qwen3.5-397B 做 raw-frame 标注和 candidate selection。Hand recon crop 有一点帮助，但提升只有 `+0.42pp` fixed-boundary accuracy，不值得为所有数据强行跑；更适合在已经有 hand asset 的数据集里作为辅助候选输入。
