# UTTT Notes

## 一 Installation

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


## 七 MPS patch

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


