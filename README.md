# MP3D ObjectNav 统一评测器

一个基于 Habitat-Lab 的轻量 ObjectNav 评测器，用于在相同协议下比较不同
Agent。评测器只负责环境、episode、观测过滤、动作校验、指标和结果保存；目标
识别、建图、探索、规划与策略全部属于 Agent，不写进评测器。

## 已验证环境

- Python 3.9
- reference track：Habitat-Lab / Habitat-Sim 0.2.4
- compatibility track：Habitat-Lab / Habitat-Sim 0.3.3
- MP3D ObjectNav v1 episodes
- Matterport3D 场景数据

两个版本都已用真实 MP3D `val_mini` episode 验证 CLI、观测、动作、指标和输出。
正式结果注明 Habitat-Lab 与
Habitat-Sim 版本，不要把 0.2.4 和 0.3.3 的结果混在同一排名中。

Habitat 和 MP3D 不会由本项目自动安装或分发，以免覆盖使用者已有的仿真环境。

## 安装

先准备一个能独立运行 Habitat 的 Conda 环境，再安装评测器。已有 Habitat 0.2.4
环境时可直接跳到克隆本项目这一步。

新建 Habitat 0.3.3 环境的实测命令：

```bash
conda create -n objectnav-eval-h033 python=3.9 \
  habitat-sim=0.3.3 headless -c conda-forge -c aihabitat
conda activate objectnav-eval-h033

git clone --branch v0.3.3 --depth 1 \
  https://github.com/facebookresearch/habitat-lab.git habitat-lab-v0.3.3
python -m pip install -e habitat-lab-v0.3.3/habitat-lab
python -m pip check
```

然后安装评测器：

```bash
git clone git@github.com:vvv257/mp3d-objectnav-eval.git
cd mp3d-objectnav-eval
python -m pip install -e ".[test]"
pytest -q
```

建议 0.2.4 和 0.3.3 使用不同 Conda 环境，不要在原环境中直接升级 Habitat。

## 数据路径

默认配置使用 Habitat 常见目录结构：

```text
data/
  datasets/objectnav/mp3d/v1/val/val.json.gz
  datasets/objectnav/mp3d/v1/val/content/*.json.gz
  datasets/objectnav/mp3d/v1/val_mini/val_mini.json.gz
  datasets/objectnav/mp3d/v1/val_mini/content/*.json.gz
  scene_datasets/mp3d/<scene-id>/<scene-id>.glb
```

如果数据位于其他目录，复制一份本地配置并修改 `data_path`、`scenes_dir`：

```bash
cp configs/mp3d_objectnav_val_mini.yaml configs/mp3d_objectnav_val_mini.local.yaml
cp configs/mp3d_objectnav_v1.yaml configs/mp3d_objectnav_v1.local.yaml
```

`*.local.yaml`、`data/`、`results/` 和模型权重默认不会提交到 Git。

## 接入 Agent

Agent 通过 `模块:类或工厂` 导入，提供 `reset(episode)`、`act(observation)` 即可。
原有 `ObjectNavAgent` 继承方式继续有效：

```python
from typing import Any, Mapping

from objectnav_eval import ObjectNavAgent
from objectnav_eval.agent import EpisodeInfo


class MyAgent(ObjectNavAgent):
    def reset(self, episode: EpisodeInfo) -> None:
        self.state = None

    def act(self, observation: Mapping[str, Any]) -> str:
        return "move_forward"
```

默认提供给 Agent 的观测为 `rgb`、`depth`、`gps`、`compass`、`objectgoal`。
评测器不会把 Habitat 环境、目标坐标、goal viewpoints、navmesh 或模拟器真值传给
Agent。

允许的动作：`stop`、`move_forward`、`turn_left`、`turn_right`、`look_up`、
`look_down`。

### 参数化动作与已有 Agent

`act()` 也可以返回以下命令：

```python
{"type": "primitive", "action": "stop"}
{"type": "rotate", "degrees": 90}  # 正数左转，负数右转
{"type": "move", "meters": 1.0}   # 只允许正数前进
```

距离和角度按实际 Habitat 步长量化：`floor(abs(request) / step_size + 0.5)`。
每条命令最多 64 个基础动作，每个基础动作都计入 `num_steps`。非有限值、零命令、
量化为零步、超长命令会报错；episode 结束立即取消剩余动作。执行期间不再次调用
Agent；距离表示名义前进量，碰撞后不会自动补走。闭环控制请每步返回一个基础动作。

整数 ID 或枚举需要明确映射。例如 Agent 的 `3` 表示左转：

```yaml
type: discrete
mapping:
  3: turn_left
  9: stop
```

将其保存为 `my_adapter.yaml`，在原 CLI 命令后加 `--adapter-config my_adapter.yaml`。
完整示例见 `configs/adapters/discrete.yaml`（其中 ID 仅为示例，请按 Agent 定义修改）。
`configs/adapters/motion.yaml` 可将 `{"degrees": 90}`、`{"meters": 1}` 转换为标准命令。
已经输出标准字符串或标准命令的 Agent 不需要指定 Adapter。

Adapter 中可选 `observations` 按传感器名配置 `rename`、`resize: [H, W]`（最近邻）、
`transpose`、`scale`、`offset`、`clip`、`dtype`。顺序是 resize → scale/offset → clip →
transpose → dtype；深度单位转换参数由使用者根据运行元数据中的传感器配置明确填写。
转换输入仅包含任务允许且本次开放的观测。自定义 `predict()` 等接口请自己写一个
具有 `reset/act` 的包装类，可复用 `objectnav_eval.adapters.BaseAdapter`，不会自动猜测签名。

### 任务协议

旧配置自动使用 `mp3d_objectnav_v1`。新配置可增加
`task_protocol: tasks/mp3d_objectnav_v1.yaml`；文件路径相对评测配置所在目录。
协议定义动作、观测、必需指标、Habitat 配置与 episode 身份策略。
`adapter_config` 也可写入评测配置，路径同样相对该文件；CLI 参数路径相对当前目录。
启动时检查环境是否提供协议所需的动作、观测和指标。
`strict_actions` 仅保留配置兼容性，设为 false 也不能绕开任务动作校验。

新增任务使用独立协议 YAML，`episode_identity` 可以是 `模块:函数`，返回稳定字符串。
ObjectNav 沿用原有 UID。评测器还会拒绝使用与 backend 已校验协议不一致的协议，
避免默认 ObjectNav 协议套到别的任务环境上。

### R2R 指令导航协议（vln_ce_r2r_v1）

`configs/tasks/vln_ce_r2r.yaml` 定义 VLN-CE R2R 协议，等价的内置写法是
`task_protocol: vln_ce_r2r_v1`。该协议只描述 Habitat-Lab 0.2.4/0.3.3
`benchmark/nav/vln_r2r.yaml` 实际组合出来的任务：

- 动作是 `stop`、`move_forward`、`turn_left`、`turn_right`（`VLN-v0` 任务的动作空间，
  没有 `look_up`、`look_down`），不照搬 ObjectNav 的六种动作。
- 观测是 `rgb`、`depth`、`instruction`，其中 `instruction` 是 Habitat
  `InstructionSensor` 给出的 `{text, tokens, trajectory_id}` 映射。
- 必需指标是该配置实际提供的 `success`、`spl`、`distance_to_goal`；0.2.4 的 VLN 配置
  不提供 `ndtw`，因此不写进协议。
- Episode 身份用场景、指令文本、`trajectory_id`、起点位姿和目标
  共同生成（`vln_ce_episode_uid`），与 ObjectNav 的 UID 策略互不共用。
  数据集在原路径被替换且物理 episode 改变时，断点续跑会因 UID 不同而拒绝。
- 运动参数与成功判据都从生效的 Habitat 配置读取后写入 `run_metadata.json`：
  VLN-CE 的 `turn_angle` 是 15 度（不是 ObjectNav 的 30 度），`forward_step_size` 为
  `0.25`，`success_criteria` 记录 `distance_to` 与 `success_distance`。

`configs/mp3d_vln_r2r.yaml` 是公开评测配置模板。复制为 `*.local.yaml` 并将数据、
场景路径改成本机路径后，可以用不继承评测器基类的最小 Agent 验收语言观测接入：

```bash
cp configs/mp3d_vln_r2r.yaml configs/mp3d_vln_r2r.local.yaml
# 编辑 local 文件中的 data_path 和 scenes_dir
objectnav-eval \
  --agent examples.instruction_smoke_agent:InstructionSmokeAgent \
  --config configs/mp3d_vln_r2r.local.yaml \
  --output results/r2r_instruction_smoke \
  --num-episodes 1
```

`InstructionSmokeAgent` 会明确读取 `instruction.text`，前进两步后停止。它只验证 Agent
接口、观测权限、动作和记录链路，不是指令跟随算法。已使用真实 R2R train episode
和 Matterport3D 场景在 Habitat 0.2.4、0.3.3 各运行一次，两个版本的 Episode 身份、
动作数和指标一致。

没有官方 R2R 数据时，仍可使用合成数据集成测试检查协议接线。
`tests/integration/vln_ce_fixture.py` 用 ObjectNav episode 的真实起点与目标物体视点、
真实 MatterPort3D 场景和导航网格生成合成 VLN 数据集，指令文本是占位内容。
真实 VLN-v0 环境上已验证语言观测能到达 Agent、观测权限过滤、Episode 身份、
运动参数与指标读取、离散与运动命令轨迹一致。合成数据上的运行不是 VLN-CE 基准结果，
不得与 R2R/VLN-CE 的排行榜数字比较。运行方式：

```bash
OBJECTNAV_TEST_VLN_SCENES_DIR=/path/to/MatterPort3D \
OBJECTNAV_TEST_VLN_EPISODE_SOURCE=/path/to/objectnav/mp3d/v1/val_mini/val_mini.json.gz \
  pytest -q tests/integration/test_vln_ce_task.py
```

上述验证证明“指令导航可以规范接入统一评测器”，不等于已完整复刻历史
VLN-CE 排行榜协议。当前 Habitat 内置配置不提供 nDTW，成功距离等语义以
`run_metadata.json` 中实际生效配置为准，不得直接与其他 VLN-CE 协议的数字混排。

## 运行

先用 RandomAgent 做 smoke test：

```bash
objectnav-eval \
  --agent examples.random_agent:RandomAgent \
  --config configs/mp3d_objectnav_val_mini.local.yaml \
  --output results/random_val_mini_smoke \
  --num-episodes 1
```

评测其他方法只需替换 Agent entrypoint：

```bash
objectnav-eval \
  --agent my_method.agent:MyObjectNavAgent \
  --config configs/mp3d_objectnav_v1.local.yaml \
  --output results/my_method_val
```

省略 `--num-episodes` 时会按确定性顺序评测整个 split。

### 中断后会怎样

每完成一个 episode 就会保存结果并显示进度。Agent 或环境报错、`Ctrl+C`
中断时，评测会立即停止，已完成的 episode 不会丢失，错误记录在
`errors.jsonl`。排除问题后，在原命令末尾加 `--resume` 即可从下一个
episode 继续。续跑前会校验 Agent 入口及代码指纹、配置、评测器代码版本、
Habitat 版本和已完成
episode 的身份；不一致时会拒绝续跑。不加 `--resume` 也不会覆盖已有结果。
任务协议、Adapter 代码/配置、动作转换器版本或生效 Habitat 配置变化也会拒绝续跑。
旧结果文件仍可读取，但升级评测器后不能直接继续旧版本运行；请用原代码版本续跑。
如果进程被 `kill -9` 或机器断电，可能没有当次错误记录，但之前已完成的
episode 仍可续跑。

## 指标

指标直接读取 `Habitat.Env.get_metrics()`，不在本项目中重新实现：

- `success`：调用 `stop` 且距离目标小于成功阈值。
- `spl`：Success weighted by Path Length。
- `soft_spl`：使用连续进度替代二值成功的 SPL。
- `distance_to_goal`：到最近有效目标的 Habitat geodesic distance。

正式比较时，必须保持 Habitat 版本、数据 split、episode 顺序、传感器、动作空间、
步数上限和成功距离一致。

## 输出

每次运行生成：

```text
episodes.jsonl         每个 episode 的完整记录
episodes.csv           扁平表格
summary.json           指标均值、episode 数和平均步数
run_metadata.json      版本、Git 提交、配置哈希、overrides 和平台信息
evaluator_config.yaml  本次运行使用的原始配置
```

新增 `primitive_action_count`、`agent_command_count`、`task_id`、`protocol_version`。
`episodes.jsonl` 中的 `action_conversions` 记录运动命令请求值、名义量化值、
误差（量化值减请求值）、计划步数与实际执行步数；`action_conversion_error` 为量化误差列表。
转换失败则在 `errors.jsonl` 的同名字段记录错误原因与已执行步数。
元数据保存任务与 Adapter 配置、代码指纹、观测转换规则、实际动作参数和生效配置哈希。
代码指纹不能覆盖所有动态导入、依赖与权重，因此 `reproducibility_complete` 明确为 false；
完整复现还需使用者提供下述方法依赖和模型记录。

单元测试：`pytest -q`（没有数据时真实仿真测试默认跳过）。真实验收：

```bash
OBJECTNAV_TEST_CONFIG=configs/mp3d_objectnav_val_mini.local.yaml \
  pytest -q tests/integration
```

该测试检查离散/运动命令逐步观测与指标一致、中断续跑、Adapter 改动拒绝续跑。
VLN-CE 的合成数据集成测试见上文“VLN-CE 协议”一节，需另外两个环境变量开启。

发布结果时，还应保存被评测方法的代码提交、依赖锁定文件、模型与配置哈希、随机种子
和硬件信息。详细公平性约束见 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)。
