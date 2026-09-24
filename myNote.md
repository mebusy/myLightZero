# UTTT Notes

## 0. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

PyTorch 官方目前支持 macOS 的 pip 安装；安装后先确认 MPS 是否可用。

```bash
python - <<'PY'
import platform
import torch

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("device:", "mps" if torch.backends.mps.is_available() else "cpu")
PY
```

如果后续运行遇到某个 MPS 未实现算子，可临时：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

这会让不支持的 MPS 算子退回 CPU；但它适合调试/兼容性，不是长期最高性能方案。


## 一 MPS patch

针对 UTTT + train_alphazero 的最小改动

文件 | 是否需要改 | 原因
--- | --- | ---
lzero/entry/train_alphazero.py | 必须 | 你执行 config 时实际经过的训练入口；当前只选 cuda 或 cpu。
lzero/entry/eval_alphazero.py | 建议 | 以后单独加载 checkpoint 做评估时需要，否则评估会落到 CPU。
lzero/agent/alphazero.py | 仅可选 | 仅当你写 Python 代码用 AlphaZeroAgent(...).train()，而非调用 train_alphazero() 时经过。
UTTT config | 必须 | 提供 mps=True / cuda=False 配置。
其他 MuZero / UniZero / async / DDP 文件 | 不需要 | 当前 AlphaZero 训练路径不会 import 或执行它们。
tests/* | 不需要 | 测试里强制写 cuda 是测试设定，不影响你的训练。
loss_landscape/* | 不需要 | 不是训练 pipeline。
rnd_reward_model.py | 不需要 | AlphaZero 不经过 RND reward model。


所以，严格说：

- 只训练：改 1 个框架文件 + 1 个你的 config 文件；
- 训练并单独评估 checkpoint：再加 eval_alphazero.py，即 2 个框架文件 + config；
- 还要使用 AlphaZeroAgent API：再改 agent/alphazero.py。




## 二、目标目录结构

```bash
LightZero/
└── zoo/
    └── board_games/
        └── ultimate_tictactoe/
            ├── __init__.py
            ├── envs/
            │   ├── __init__.py
            │   └── ultimate_tictactoe_env.py
            ├── config/
            │   ├── __init__.py
            │   └── ultimate_tictactoe_alphazero_sp_config.py
            └── tests/
                └── test_ultimate_tictactoe_env.py
```


文件 | 必要性 | 作用
--- | --- | ---
ultimate_tictactoe_env.py | 必须 | UTTT 规则、81 个动作、合法着、终局、观测
ultimate_tictactoe_alphazero_sp_config.py | 必须 | self-play、网络、MCTS、replay buffer 参数
test_ultimate_tictactoe_env.py | 强烈建议 | 验证“送盘”、小盘完成、自由落子、终局等

> LightZero 的原始 AlphaZero TicTacToe config 使用 episode_alphazero collector、AlphaZero policy 和 subprocess 环境管理器；这正好是 UTTT self-play 所需的 pipeline。


## 三、环境设计：不要把 UTTT 当成“普通 9×9 井字棋”

UTTT 的 board 可以物理存为：

```bash
board:       int8[9, 9]    # 0 empty, 1 X, 2 O
sub_status:  int8[3, 3]    # 0 open, 1 X won, 2 O won, 3 draw/closed
forced_board: int          # 0..8 或 -1（自由落子）
current_player: 1 or 2
```

actino:

```bash
action = row * 9 + col,  action ∈ [0, 80]
row = action // 9
col = action % 9
```

## 四、观测编码：推荐 6 个 9×9 plane

UTTT 应明确改为 81 个动作和至少 (6, 9, 9) 的观测。原模板将 model 的 observation_shape 与 action_space_size 放在 config 内，因此不必改 LightZero 的 AlphaZero 模型本身

我建议采用 STM 相对视角，和你前面讨论的 NNUE STM/NSTM 思想一致：

Channel | 内容
0 | 当前走子方（STM）在 9×9 棋盘上的子
1 | 对手（NSTM）在 9×9 棋盘上的子
2 | 当前合法落子 mask
3 | STM 已赢的小棋盘，每个已赢 mini-board 对应的 3×3 区域全置 1
4 | NSTM 已赢的小棋盘
5 | 已关闭但无人获胜的 mini-board（draw/full）

这里最重要的是 channel 2：

- MCTS 最终仍依赖 action_mask 来屏蔽非法动作；
- 但将 forced-board / legal region 同时放进 observation，有助于网络理解“我被送到哪里”；
- 只用棋子平面而不显式表达 mini-board 完结状态，会让网络自己从 9 格局部推导大量规则状态，学习更慢。


## 五、核心环境文件

推荐复制 zoo/board_games/gomoku/envs/gomoku_env.py 的 BaseEnv、reset、step、simulate_action、`create_*_env_cfg` 框架，然后替换棋盘规则。Gomoku 环境本身已经采用了“当前方/对方”的相对观测以及 action_mask。

<details>
<summary>
ultimate_tictactoe/envs/ultimate_tictactoe_env.py
</summary>

```python
import copy
from typing import List

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from easydict import EasyDict


EMPTY, X, O = 0, 1, 2
DRAW = 3
BOARD_SIZE = 9
MACRO_SIZE = 3
ACTION_SIZE = 81


def winner_3x3(grid: np.ndarray) -> int:
    """Return X/O if a 3x3 grid is won; otherwise EMPTY."""
    for player in (X, O):
        if np.any(np.all(grid == player, axis=0)):
            return player
        if np.any(np.all(grid == player, axis=1)):
            return player
        if np.all(np.diag(grid) == player):
            return player
        if np.all(np.diag(np.fliplr(grid)) == player):
            return player
    return EMPTY


@ENV_REGISTRY.register("ultimate_tictactoe")
class UltimateTicTacToeEnv(BaseEnv):

    config = dict(
        env_id="UltimateTicTacToe",
        battle_mode="self_play_mode",
        channel_last=False,
        scale=True,
        collector_env_num=8,
        evaluator_env_num=2,
        n_evaluator_episode=2,
        manager=dict(shared_memory=False),
        alphazero_mcts_ctree=False,
        stop_value=1,
    )

    @classmethod
    def default_config(cls):
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    def __init__(self, cfg=None):
        self._cfg = self.default_config()
        self._cfg.update(cfg)
        self.players = [X, O]
        self.total_num_actions = ACTION_SIZE
        self.battle_mode = self._cfg.battle_mode
        self.channel_last = self._cfg.channel_last
        self.scale = self._cfg.scale
        self.alphazero_mcts_ctree = self._cfg.alphazero_mcts_ctree
        self._env = self

    def reset(self, start_player_index=0, init_state=None, **kwargs):
        self.start_player_index = start_player_index
        self._current_player = self.players[start_player_index]

        if init_state is None:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            self.sub_status = np.zeros((MACRO_SIZE, MACRO_SIZE), dtype=np.int8)
            self.forced_board = -1
        else:
            # 建议把 board/sub_status/forced_board/current_player 序列化为 dict；
            # 首版也可以仅支持从头 reset。
            self._restore_state(init_state)

        self._observation_space = gym.spaces.Box(
            low=0, high=1, shape=(6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(ACTION_SIZE)
        self._reward_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        return self._make_obs()

    @property
    def current_player(self):
        return self._current_player

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @property
    def next_player(self):
        return O if self.current_player == X else X

    @property
    def legal_actions(self) -> List[int]:
        open_boards = self.sub_status == EMPTY

        # 被强制的小盘仍开放：只能走该小盘的空格。
        if self.forced_board != -1:
            br, bc = divmod(self.forced_board, MACRO_SIZE)
            if open_boards[br, bc]:
                return self._empty_actions_in_subboard(br, bc)

        # 否则自由走：任何尚未结束的小盘内的空格。
        legal = []
        for br in range(MACRO_SIZE):
            for bc in range(MACRO_SIZE):
                if open_boards[br, bc]:
                    legal.extend(self._empty_actions_in_subboard(br, bc))
        return legal

    def _empty_actions_in_subboard(self, br, bc):
        r0, c0 = br * 3, bc * 3
        sub = self.board[r0:r0 + 3, c0:c0 + 3]
        rows, cols = np.where(sub == EMPTY)
        return [(r0 + r) * BOARD_SIZE + (c0 + c) for r, c in zip(rows, cols)]

    def step(self, action):
        if action not in self.legal_actions:
            raise ValueError(f"illegal UTTT action={action}")

        row, col = divmod(int(action), BOARD_SIZE)
        player_just_moved = self.current_player
        self.board[row, col] = player_just_moved

        # 更新刚刚落子所在的 mini-board。
        macro_r, macro_c = row // 3, col // 3
        self._update_subboard_status(macro_r, macro_c)

        done, winner = self.get_done_winner()

        # action 的局部坐标决定对手被送往哪个小盘。
        target_r, target_c = row % 3, col % 3
        if self.sub_status[target_r, target_c] == EMPTY:
            self.forced_board = target_r * 3 + target_c
        else:
            self.forced_board = -1

        self.current_player = self.next_player

        reward = np.array(float(done and winner == player_just_moved), dtype=np.float32)
        info = {"winner": winner, "next player to play": self.current_player}
        if done:
            # evaluator 常用的从 player-1/X 视角回报
            info["eval_episode_return"] = (
                1.0 if winner == X else -1.0 if winner == O else 0.0
            )

        return BaseEnvTimestep(self._make_obs(), reward, done, info)

    def _update_subboard_status(self, br, bc):
        if self.sub_status[br, bc] != EMPTY:
            return
        r0, c0 = br * 3, bc * 3
        sub = self.board[r0:r0 + 3, c0:c0 + 3]
        winner = winner_3x3(sub)
        if winner != EMPTY:
            self.sub_status[br, bc] = winner
        elif not np.any(sub == EMPTY):
            self.sub_status[br, bc] = DRAW

    def get_done_winner(self):
        macro = np.where(
            self.sub_status == X, X,
            np.where(self.sub_status == O, O, EMPTY)
        )
        winner = winner_3x3(macro)
        if winner != EMPTY:
            return True, winner
        if not self.legal_actions:
            return True, DRAW
        return False, EMPTY

    def current_state(self):
        stm = self.current_player
        nstm = self.next_player

        state = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        state[0] = (self.board == stm)
        state[1] = (self.board == nstm)

        for action in self.legal_actions:
            r, c = divmod(action, BOARD_SIZE)
            state[2, r, c] = 1.0

        for br in range(3):
            for bc in range(3):
                r0, c0 = br * 3, bc * 3
                status = self.sub_status[br, bc]
                if status == stm:
                    state[3, r0:r0 + 3, c0:c0 + 3] = 1.0
                elif status == nstm:
                    state[4, r0:r0 + 3, c0:c0 + 3] = 1.0
                elif status == DRAW:
                    state[5, r0:r0 + 3, c0:c0 + 3] = 1.0

        return state, state

    def _make_obs(self):
        action_mask = np.zeros(ACTION_SIZE, dtype=np.int8)
        action_mask[self.legal_actions] = 1
        return {
            "observation": self.current_state()[1],
            "action_mask": action_mask,
            "board": self._serialize_state(),
            "current_player_index": 0 if self.current_player == X else 1,
            "to_play": self.current_player,
        }

    def _serialize_state(self):
        return {
            "board": self.board.copy(),
            "sub_status": self.sub_status.copy(),
            "forced_board": self.forced_board,
            "current_player": self.current_player,
        }

    def _restore_state(self, state):
        self.board = np.asarray(state["board"], dtype=np.int8).copy()
        self.sub_status = np.asarray(state["sub_status"], dtype=np.int8).copy()
        self.forced_board = int(state["forced_board"])
        self.current_player = int(state["current_player"])

    def simulate_action(self, action):
        env = copy.deepcopy(self)
        env.step(action)
        return env

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def create_collector_env_cfg(cfg: dict):
        n = cfg.pop("collector_env_num")
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(n)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict):
        n = cfg.pop("evaluator_env_num")
        cfg = copy.deepcopy(cfg)
        cfg.battle_mode = "self_play_mode"
        return [cfg for _ in range(n)]

```
</details>


一个实现细节：simulate_action 必须正确
MCTS 会高频复制/重置/推进环境。因此不要在 step() 中做：

- matplotlib / pygame render；
- 日志打印；
- 巨大 Python object 分配；
- 复杂 deep-copy 的缓存对象；
- 慢速规则推导。

先写对，再优化。初版用 Python + NumPy 足够验证训练闭环；性能不足时，优先将：

legal_actions
- 小盘胜负判定
- macro 胜负判定
- 改为 Cython、Numba 或 C++ 扩展。

## 六、AlphaZero self-play config

先从 轻量网络 + 较少 simulations 开始，而不要上来就 200/800 simulations。M4 上先跑通、检查学习曲线，再逐步增大。


<details>
<summary>
ultimate_tictactoe/config/ultimate_tictactoe_alphazero_sp_config.py
</summary>

```python
from easydict import EasyDict

# Mac mini M4：先从保守配置开始。
collector_env_num = 6
evaluator_env_num = 2
n_episode = 6

num_simulations = 64
update_per_collect = 32
batch_size = 128
max_env_step = int(2_000_000)

# 首次跑通用 False；完成 MPS patch 后改为 True。
use_cuda = False
mcts_ctree = False

ultimate_tictactoe_alphazero_config = dict(
    exp_name="data_az/uttt_alphazero_m4_seed0",
    env=dict(
        env_id="UltimateTicTacToe",
        battle_mode="self_play_mode",
        channel_last=False,
        scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False),

        # 纯 self-play 不应混入随机/规则 bot。
        stop_value=1,
        alphazero_mcts_ctree=mcts_ctree,
    ),
    policy=dict(
        mcts_ctree=mcts_ctree,
        simulation_env_id="ultimate_tictactoe",
        simulation_env_config_type="self_play",

        model=dict(
            observation_shape=(6, 9, 9),
            action_space_size=81,

            # 起步模型；过拟合/价值不稳定再调宽。
            num_res_blocks=4,
            num_channels=64,
            value_head_hidden_channels=[64],
            policy_head_hidden_channels=[64],
        ),

        # LightZero 将这个字段理解为 CUDA 开关。
        # 若未 patch MPS：保持 False，走 CPU。
        cuda=use_cuda,

        board_size=9,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type="Adam",
        learning_rate=1e-3,
        grad_clip_value=1.0,

        value_weight=1.0,
        entropy_weight=0.0,

        n_episode=n_episode,
        eval_freq=10_000,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,

        mcts=dict(
            num_simulations=num_simulations,
        ),
    ),
)

main_config = EasyDict(ultimate_tictactoe_alphazero_config)

create_config = EasyDict(dict(
    env=dict(
        type="ultimate_tictactoe",
        import_names=[
            "zoo.board_games.ultimate_tictactoe.envs.ultimate_tictactoe_env",
        ],
    ),
    # macOS 初期调试建议 base；稳定后再试 subprocess。
    env_manager=dict(type="base"),

    policy=dict(
        type="alphazero",
        import_names=["lzero.policy.alphazero"],
    ),
    collector=dict(
        type="episode_alphazero",
        import_names=["lzero.worker.alphazero_collector"],
    ),
    evaluator=dict(
        type="alphazero",
        import_names=["lzero.worker.alphazero_evaluator"],
    ),
))

if __name__ == "__main__":
    from lzero.entry import train_alphazero
    train_alphazero(
        [main_config, create_config],
        seed=0,
        max_env_step=max_env_step,
    )

```
</details>

原始模板里同样设置了 model.observation_shape、action_space_size、MCTS simulations、collector 数和 batch size；这里只是把它们扩展为 UTTT 的 6×9×9 与 81 动作

运行：

```bash
python -u zoo/board_games/ultimate_tictactoe/config/ultimate_tictactoe_alphazero_sp_config.py
```

TensorBoard：

```bash
tensorboard --logdir data_az/uttt_alphazero_m4_seed0/log
```


## 八、先写这些规则测试，否则 self-play 数据没有意义

UTTT 最常见 bug 不在网络，而在规则实现。至少覆盖：

<details>
<summary>
ultimate_tictactoe/tests/test_ultimate_tictactoe_env.py
</summary>

```python
import numpy as np
from zoo.board_games.ultimate_tictactoe.envs.ultimate_tictactoe_env import (
    UltimateTicTacToeEnv,
    X,
    O,
    EMPTY,
)


def make_env():
    env = UltimateTicTacToeEnv(UltimateTicTacToeEnv.default_config())
    env.reset()
    return env


def test_first_move_can_go_anywhere():
    env = make_env()
    assert len(env.legal_actions) == 81


def test_move_sends_opponent_to_matching_local_board():
    env = make_env()

    # 全局 (row=1, col=2) 即 action=11；其局部位置是 (1, 2)，
    # 所以对手必须进入 macro board #5（行1、列2）。
    env.step(1 * 9 + 2)

    expected = set()
    for r in range(3, 6):
        for c in range(6, 9):
            expected.add(r * 9 + c)

    assert set(env.legal_actions) == expected


def test_closed_target_board_allows_free_move():
    env = make_env()

    # 目标小盘 (1, 2) 标为关闭；forced_board 指到它时必须变为自由落子。
    env.sub_status[1, 2] = X
    env.forced_board = 5

    legal = env.legal_actions
    assert len(legal) == 72  # 9x9 - 那个关闭的 3x3 小盘


def test_macro_win_ends_game():
    env = make_env()

    # X 已赢 macro board 顶行三个小盘。
    env.sub_status[0, :] = X

    done, winner = env.get_done_winner()
    assert done is True
    assert winner == X

```
</details>

还要增加：

- 小盘横、竖、两条对角线胜利；
- 小盘填满无胜方时标为 DRAW；
- 送往已关闭小盘时 forced_board=-1；
- 大盘胜利后即使别处仍有空格也必须终局；
- action_mask 中的 1 与 legal_actions 完全一致；
- clone() / simulate_action() 不修改父节点。

## 九、训练参数如何逐阶段加大

不要一开始就追求极强棋力。建议：

阶段 | 网络 | simulations | collector env | batch | 用途
--- | --- | --- | --- | --- | ---
规则验证 | 2 blocks × 32 ch | 16 | 2 | 32 | 查崩溃、查非法着
初始学习 | 4 × 64 | 64 | 6 | 128 | 看 policy/value 是否学习
正式 self-play | 6–8 × 96/128 | 128–256 | 6–10 | 128–256 | 提升棋力
强化评估 | 8 × 128 | 256–512 | 较少 | 256 | 仅候选模型对战


在 M4 机器上，瓶颈很可能先是 Python 环境模拟/MCTS tree traversal，而不是 CNN。不要只盯着 MPS；先用 profiler 确认时间到底在哪一项:

- policy forward；
- copy.deepcopy；
- legal-action 枚举；
- Python MCTS；
- env subprocess IPC；

我建议起步时使用：

```bash
env_manager = base
mcts_ctree = False
num_simulations = 16
collector_env_num = 2
```

因为这能让错误栈最可读。稳定后再试 subprocess 和 mcts_ctree=True。
LightZero 的模板默认确实存在 C++ MCTS tree 选项和 subprocess manager，但 UTTT 环境的序列化、reset、clone 都正确之前，先不要引入两层调试复杂度。


## 十、评价不应只看 self-play reward

纯 self-play 很容易出现“双方一起退化/互相适应”。至少要维护：

1. 固定随机策略；
2. 简单规则 bot：
    - 若能立即赢 mini-board，优先；
    - 若能阻止对手赢 mini-board，优先；
    - 对 macro board 的立即胜利/阻挡；
3. 历史 checkpoint 池；
4. candidate vs current-best 的交换先后手比赛。

由于 UTTT 有明显先手优势，评估一组模型时务必：

```bash
A 执 X vs B 执 O
A 执 O vs B 执 X
```

各跑同样数量，汇报：

```bash
X 胜 / O 胜 / 和棋
以及按角色交换后的总得分
```

## 十一、和你前面 NNUE 目标的关系

LightZero 的 AlphaZero CNN 很适合做：

```bash
UTTT 规则验证
+ self-play 数据生成
+ policy/value baseline
+ 研究 input planes、MCTS、温度和搜索强度
```


但它不是文章中极限 CPU NNUE 的直接训练器：


LightZero AlphaZero | 文章的最终 CodinGame bot
CNN / ResNet | 稀疏特征 FC/NNUE
完整 forward | accumulator 增量更新
policy + value | 主要 value evaluator
常规 PUCT/MCTS | jacekmax 风格展开
PyTorch/MPS | C + SIMD + 整数量化

所以最合适的路线是：

1. 用 LightZero 的 AlphaZero 先证明 UTTT 环境和 self-play pipeline 正确；
2. 保存 replay / game segments 中的 (state, MCTS visit policy, terminal outcome)；
3. 另写 PyTorch NNUE trainer，把 STM/NSTM 特征训练为 value network；
4. 最后将 NNUE 量化导出到 C/C++ 搜索器。


