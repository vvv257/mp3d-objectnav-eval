# MP3D ObjectNav 统一评测器

一个基于 Habitat-Lab 的轻量 ObjectNav 评测器，用于在相同协议下比较不同
Agent。评测器只负责环境、episode、观测过滤、动作校验、指标和结果保存；目标
识别、建图、探索、规划与策略全部属于 Agent，不写进评测器。

当前版本是可复现的 MVP。`val_mini` 只用于检查安装是否可用，不能作为正式
benchmark 结果。

## 参考环境

- Python 3.9
- Habitat-Lab 0.2.4
- Habitat-Sim 0.2.4
- MP3D ObjectNav v1 episodes
- Matterport3D 场景数据

Habitat 和 MP3D 不会由本项目自动安装或分发。

## 安装

先进入已经能运行 Habitat 0.2.4 的 Conda 环境，再安装评测器：

```bash
git clone git@github.com:vvv257/mp3d-objectnav-eval.git
cd mp3d-objectnav-eval
conda activate objectnav-eval
python -m pip install -e ".[test]"
pytest -q
```

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

Agent 必须继承 `ObjectNavAgent`，并能通过 `模块:类或工厂` 导入：

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

发布结果时，还应保存被评测方法的代码提交、依赖锁定文件、模型与配置哈希、随机种子
和硬件信息。详细公平性约束见 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)。
