# FastFill workflow：给机器人任务用的家具摆放

输入一个房间（任意多边形，有门窗）和上游给出的家具清单、必须家具、约束，输出每件家具的**尺寸、位置和朝向**，要求：

- **合法**：不碰撞、不出界、不挡门，约束满足；
- **机器人可用**：门与门之间连通，任务要用的家具机器人走得到，要打开的柜门能打开，台面上的物品在手臂范围内；
- **尺寸不改**：所有物体（必须家具和扩充家具）都保持 FastFill 预测的尺寸，流程里没有任何一步会缩放物体。
- **必须家具一定达标**：必须家具不删除，最后必须合法且可达；其余的扩充家具也尽量合法、可达，只有在全屋重新找位置也不行时，才会为了给必须家具让路而删除。

无论如何都会交出一个房间：如果必须家具仍有问题，或者流程中途出错，就交出**只放必须家具**的房间（兜底）。

技术细节（每一步怎么做、机器人检查怎么算）见 [`docs/pipeline.md`](docs/pipeline.md)（英文）。

## 流程

```
上游输入 ──① FastFill 尺寸（GPU）──② 摆放（GPT + 程序检查和修复）──> 布局
```

第 ② 步内部：
1. **规划**：调用一次 GPT，确定支撑、分组和关系；
2. **初稿**：调用两次 GPT，各给一份全屋摆放（尺寸用 ① 的结果，必须家具会标出来），选问题少的那份；
3. **检查和修复**：程序检查所有硬问题（碰撞、出界、挡门、约束、机器人检查），先修必须家具，再修其余家具；
4. **让路**：挡住必须家具的扩充家具先挪到房间别处，实在不行才删除；
5. **必须家具优先**：如果必须家具还有问题，就单独摆必须家具，再把扩充家具一件件放回去；
6. **审查**：GPT 看一遍全屋，只修严重问题，而且修完不能比原来差；
7. **放不下的去掉**：扩充家具还有物理问题（碰撞、出界、没有支撑、挡门、挂墙挡窗）时，放在家具上的先挪到台面上的空位，实在放不下才删除；
8. **挂墙物体让开前方**：挂墙物体前方的地面大部分被家具占着时（按 SceneEval 可接近度的那条地面算，空地不到 55%），沿墙挪到最近一个空地至少 65% 的位置，放在它上面的东西一起挪。不能产生新问题、不能破坏关系，地上的家具不动，也不删除任何东西；
9. **兜底**：必须家具仍有问题时，交出只放必须家具的房间。这一步由程序完成，不调用 GPT。

## 目录

| 路径 | 内容 |
|---|---|
| `fastfill_workflow/workflow/` | 流程代码。入口是 `pipeline.run_draft_workflow`；任务模式在 `simple.py`；机器人检查在 `robot.py` |
| `fastfill_workflow/workflow/cli.py` | 命令行 `python -m workflow`：`sizes` 是**第 ① 步**（FastFill 算尺寸），`run` 是**第 ② 步**（摆放），`run --checkpoint` 两步一起跑 |
| `fastfill_workflow/examples/task_rooms.jsonl` | 两个示例房间：一个最小示例，一个真实房间（带任务） |
| `fastfill_workflow/*.json` | 尺寸表和规则表，运行时会读取 |
| `fastfill_workflow/tests/` | 113 个单元测试（不调用 GPT；`test_run_cli.py` 用一个很小的 FastFill 模型把整条通路跑一遍） |
| `docs/program_repair_rules.md` | 程序修复的每一条规则、阈值和代码位置 |
| `fastfill/` | 流程用到的 FastFill 代码：数据格式、合法性检查、模型加载和推理 |

## 安装

```bash
git clone https://github.com/XXXPILAR/fastfill-workflow.git && cd fastfill-workflow
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd fastfill_workflow && ../.venv/bin/python -m pytest -q tests      # 应该是 113 passed
```

下面的命令都在 `fastfill_workflow/` 目录下运行。

### API key（不要放进仓库）

- **中转站**（`--api relay`）：新建 `~/.fastfill_api.env`，权限设为 600，写入：
  ```
  OPENAI_API_KEY=<key>
  OPENAI_MODEL=gpt-6.1-sol
  ```
  地址用 `--base-url` 指定，只接受 `workflow/credentials.py` 里 `RELAY_HOSTS` 列出的地址。
- **OpenAI 官方**（`--api official`）：key 写在 `fastfill_workflow/.env` 或 `~/.fastfill_openai.env`，格式同上。

key 不会被打印，也不会写进日志。

### FastFill 模型权重

[`XXXpilar/fastfill-v4detail-1.7b-full`](https://huggingface.co/XXXpilar/fastfill-v4detail-1.7b-full)（Qwen3-1.7B 全量微调，约 6.9 GB，公开）：

```bash
# 在仓库根目录
hf download XXXpilar/fastfill-v4detail-1.7b-full --local-dir runs-hf
mkdir -p runs && mv runs-hf/main4-cell05-v4detail-e5-1.7b-all-full runs/
```

`model-step-4500/` 旁边必须有 `tokenizer/`，代码会从那里读取分词器。

模型是用 transformers 5.14.1 训练和保存的，要按 `requirements.txt` 的版本安装：transformers 4.x 读不了它的分词器（`extra_special_tokens` 的格式不同）。这组版本在 Mac（Apple MPS）上跑通过完整流程，算出的尺寸和 GH200 上的只差 0.006 mm。

## 输入：一行一个房间（`rows.jsonl`）

```json
{"schema_version": "fastfill.v2",
 "condition": {"schema_version": "fastfill.v2",
               "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0,0],[4,0],[4,3],[0,3]], "height_m": 2.6, "room_type": "bedroom",
                        "openings": [{"type": "door", "wall_index": 0, "position_on_wall": 0.5, "width_m": 0.9, "height_m": 2.0}]},
               "objects": [{"id": "obj_0000", "category": "bed", "description": "A double bed with a grey fabric headboard.", "support_parent": "floor"},
                           {"id": "obj_0001", "category": "nightstand", "description": "A small oak nightstand with one drawer.", "support_parent": "floor"},
                           {"id": "obj_0002", "category": "lamp", "description": "A white ceramic table lamp.", "support_parent": "obj_0001"}],
               "constraints": [{"type": "against_wall", "object_id": "obj_0000"}]},
 "task": {"required_ids": ["obj_0001", "obj_0002"], "articulation_ids": ["obj_0001"]}}
```

- **必填**：`schema_version`、`room.frame`、`room.floor_polygon_xy_m`、`objects`、`constraints`（可以是空列表）。字段的完整定义见 `fastfill/v2/schema.py`。
- **建议填写 `room.height_m`**：不填时用默认房高，很高的家具（比如 2.8 米的冰箱）可能会被判为顶到天花板。
- **每件家具都要有描述**，不能只写类别，否则会被拒绝。`support_parent` 可以是 `floor`、`wall`，或者另一件家具的 id（放在它上面）。
- **约束**只支持 8 种：`on`、`between`、`against_wall`、`faces`、`faces_direction`、`near`、`clearance`、`keepout`。其他类型会被拒绝。
- **`task.required_ids`**：必须家具。**`task.articulation_ids`**：要打开的家具（柜门、抽屉、冰箱门等），必须也在 `required_ids` 里。不在 `required_ids` 里的都算扩充家具。
- 不需要真实布局。

## 运行

一条命令跑完整条流程（第 ① 步 FastFill 算尺寸，第 ② 步摆放），默认参数就是论文实验的设置：

```bash
WORKFLOW_REASONING_EFFORT=medium ../.venv/bin/python -m workflow run --rows examples/task_rooms.jsonl --index 0 1 \
    --checkpoint ../runs/main4-cell05-v4detail-e5-1.7b-all-full/model-step-4500 --out my_out
```

GPU 和 API 不在同一台机器上时，分两步：

```bash
# 第 ① 步，在有 GPU 的机器上：输出和 rows 逐行对齐
../.venv/bin/python -m workflow sizes --rows examples/task_rooms.jsonl \
    --checkpoint ../runs/main4-cell05-v4detail-e5-1.7b-all-full/model-step-4500 --out my_sizes.jsonl --device cuda
# 第 ② 步，在能连上 API 的机器上
../.venv/bin/python -m workflow run --rows examples/task_rooms.jsonl --sizes-file my_sizes.jsonl --workers 4 --out my_out
```

- 每行有 `task` 时开启任务模式（必须家具、要打开的家具）；没有 `task` 的行按普通模式摆放。
- `--device auto`（默认）：有 CUDA 用 CUDA，Mac 上用 MPS，否则用 CPU。在 GH200 上，每个房间中位数 0.14 秒（不算加载模型）。
- **没有 GPU 时**：用 `--draft-sizes` 代替 `--checkpoint`，由 GPT 在初稿里自己给尺寸，后面的步骤不变。我们的对比实验里，GPT 给的尺寸误差比 FastFill 大（0.56 对 0.33）。
- 默认用中转站（`--api relay`，`--model gpt-6.1-sol`）；`--api official` 用 OpenAI 官方。`WORKFLOW_REASONING_EFFORT`：GPT 的推理强度，`medium`（默认）或 `xhigh`。API 暂时出错时会一直重试。
- `--workers 4`：同时摆 4 个房间；`--rpm 30`（默认）：所有房间合计每分钟最多 30 次请求。
- 中断后重跑同一条命令即可：已经有 `room_XXX.json` 的房间会跳过。
- 其他默认参数：机器人宽 0.5 m，规划时再留 0.1 m 余量，手臂 0.4 m；初稿 2 份；修复 3 轮，每轮程序搜索最多 180 秒（`--robot-width`、`--drafts`、`--repair-rounds`、`--repair-seconds` 可以改）。`--no-llm` 不调用 GPT（规则规划、程序摆放、不审查），只用于测试。
- 参考：test150doors 的 150 个房间（每个 6–189 件家具），每个房间的费用中位数约 $0.33，用时中位数约 3.4 分钟。房间越大越慢。

## 输出

每个房间一个 `my_out/room_XXX.json`：

| 字段 | 含义 |
|---|---|
| `status` | `clean`：没有剩下的硬问题；`best_effort`：修复后还剩硬问题（见 `diagnostics.hard_issues`）；`fallback_required_only`：兜底，只放了必须家具 |
| `layout.objects` | 每件家具的 `id`、`target_size_local_m`、`bottom_center_m`、`yaw_rad`、`support_parent`（坐标约定见下一节）。被删掉的扩充家具不在这里 |
| `diagnostics.task.required_ok` | 必须家具是否全部合法且可达 |
| `diagnostics.task.required_issues` | 必须家具还剩的问题 |
| `diagnostics.task.removed` | 删掉的扩充家具，每件都写了原因，以及放在它上面、跟着一起删掉的家具 |
| `diagnostics.task.moved` | 被挪动过的家具，每件都写了是哪一步挪的 |
| `diagnostics.task.wall_clear` | 第 8 步：挂墙物体有几件、几件前方空地不够、挪了哪些（从哪到哪、多远、有没有换墙）、哪些找不到位置 |
| `diagnostics.task.expansion_issues` | 扩充家具还剩的问题（不会因为这些问题被删除） |
| `diagnostics.unmet_constraints` | 只在兜底时有：没能满足的约束 |
| `cost_usd`、`wall_s`、`llm_calls` | 费用、用时、GPT 调用次数 |
| `sizes_from` | 尺寸来自哪里：`fastfill`、`sizes_file`、`draft`（GPT 给的）或 `catalog`（FastFill 处理不了的房间） |

另外还有：`summary.json`（每个房间的状态、用时、费用）、`sizes.jsonl`（用 `--checkpoint` 时 FastFill 算出的尺寸）、`llm_calls/room_XXX.jsonl`（每次 GPT 调用的输入和输出，不含 key）。

## 坐标约定

| | 约定 |
|---|---|
| 世界坐标 | 右手系，**Z 朝上**，单位米；地面在 `floor_z_m`（默认 0） |
| 墙 | 第 `i` 面墙是 `floor_polygon_xy_m` 的第 `i` 个顶点到第 `i+1` 个顶点（最后一个顶点连回第 0 个） |
| 门窗 | `wall_index` 指墙；`position_on_wall` 是门窗**中心**在这面墙上的位置，从第 `i` 个顶点算起，取 0–1 的比例；`width_m` 是宽度 |
| `bottom_center_m` | 家具包围盒**底面中心**的世界坐标 [x, y, z]；落地家具的 z 就是地面高度，放在别的家具上时 z 是那件家具的顶面高度 |
| `target_size_local_m` | **[深, 宽, 高]**：深是前后方向，沿家具局部 +X；宽是左右方向，沿局部 +Y；高沿 +Z |
| `yaw_rad` | 家具绕 +Z 轴的旋转角，从世界 +X 轴**逆时针**量，单位弧度。家具**正面朝向局部 +X**，也就是世界方向 (cos yaw, sin yaw)。例如 yaw = π/2 表示正面朝世界 +Y |
| 挂墙的家具 | 背面贴墙，正面朝房间里 |

正面朝向由第 ② 步的 GPT 决定。FastFill 只提供尺寸：它预测的 yaw 分不清正反（训练时按 π 取模），所以流程不用它的 yaw。

**转换成 SAGE / RoomGenBench 的约定**（正面是局部 +Y，角度用度）：`width = 宽`，`length = 深`，`yaw_sage = yaw − 90°`。

## 已知限制

- 约束只支持上面列出的 8 种。互相矛盾的约束需要上游避免；遇到时兜底仍会交出合法、可达的必须家具，并列出没满足的约束。
- 扩充家具被删掉时，和它有关的约束会一起去掉，目前没有单独报告。
- 大房间比较慢：程序修复每轮最多 180 秒，上百件家具的房间可能要十几分钟。
