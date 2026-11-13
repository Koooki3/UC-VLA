# UC-VLA — Unified Context for Long-Horizon Vision-Language-Action

> **Status: research specification and executable contracts (NOT a trained VLA).**
> As of 2026-11-11. Working title; not a claim of novelty, acceptance, or SOTA.

**Research question.** Can a robot learn a bounded, provenance-aware, *decision-sufficient context* that is updated through action, observation and verification, and improve closed-loop long-horizon control at a matched compute/memory/latency budget?

**One unified loop:** `observe → update belief → decide what to retain/retrieve → condition policy → execute → verify → reconcile belief`.

**中文定位：**这不是简单的 VLA + Memory Bank。UC-VLA 研究未来决策损失驱动的压缩、真实执行反馈驱动的上下文版本化更新，以及延迟约束下的主动检索/再观察。完整研究计划见 `docs/`。目标是做可证伪、可复现、能与近期强基线正面竞争的顶会级研究，而不是提前承诺论文接收。

## Contents

| File | Purpose |
| --- | --- |
| `docs/00_EXECUTIVE_DECISION.md` | 审查后研究定位、核心假设、go/no-go |
| `docs/01_PRIOR_ART.md` | 2026-11-11 文献审查、与最近邻工作的区别 |
| `docs/02_UNIFICATION.md` | “同一性”可操作定义与形式化不变量 |
| `docs/03_THEORY.md` | POMDP、decision regret、信息瓶颈、结论边界 |
| `docs/04_ARCHITECTURE.md` | 训练/推理/存储/版本化/实时调度架构 |
| `docs/05_LEARNING.md` | 数据构造、监督预训练、因果干预、离线/在线 RL |
| `docs/06_BENCHMARKS.md` | 数据集、benchmark、成对历史、反泄露设计 |
| `docs/07_EXPERIMENTS.md` | 预注册假设、实验矩阵、统计学、资源预算 |
| `docs/08_ADVERSARIAL_REVIEWS.md` | 五轮内部对抗性审查及对应架构修正 |
| `docs/09_IMPLEMENTATION.md` | 依赖图、开发 milestones、DoD/验收门槛 |
| `docs/10_CLUSTER.md` | GPU/SLURM、数据治理、manifest、故障恢复 |
| `docs/11_PAPER_STRATEGY.md` | 论文叙事、主张强度、关键图表、投稿门槛 |
| `docs/12_SAFETY_AND_LIMITS.md` | 真机安全、证据与结论局限 |
| `docs/13_LEVI_BRIDGE.md` | LEVI/AERI 的隔离集成策略 |
| `docs/14_AGENT_EXECUTION.md` | Coding Agent 执行总规范 |
| `docs/15_OPEN_DECISIONS.md` | 需要集群/机器人确认的可变假设 |
| `docs/17_POLICY_EVAL_CONTRACT.md` | M2 PolicyAdapter/访问边界/可复现评测契约 |
| `docs/19_SODH_V2_SPEC.md` | SODH v2 预注册研究规格（实现前冻结） |
| `docs/21_G0_POWER_REPORT.md` | G0 弱泄漏检测功效（含盲区） |
| `docs/23_V1_V2_CONSTRUCT_VALIDITY.md` | v1/v2 构念效度差异、四项记忆指标、F7/B2b |
| `docs/24_OPEN_SOURCE_READINESS.md` | 开源准备（未创建远程） |
| `docs/18_M1_ADVERSARIAL_REVIEW.md` | M1 对抗性复审：发现、修复、v2 规格 |
| `reviews/` | 复审的证伪脚本与原始输出 |
| `agents/MASTER_IMPLEMENTATION_PROMPT.md` | 给 Codex/Claude Code 的直接执行 Prompt |
| `paper/outline.md` | 英文论文草案提纲 |
| `src/ucvla/` | 无外部 ML 依赖：contracts、toy、M1 episode 数据底座/SODH/splits/audit/symbolic readers |
| `tests/` | 防未来泄露、版本控制、因果玩具任务的单元测试 |
| `configs/` | 可复制的预注册实验模板（非真实配置） |
| `scripts/` | doctor、dry-run、SLURM 模板 |

## Day-0 smoke test (no GPU, no network, no robot)

```bash
# Python 3.11-3.13 required (system 3.14 is unsupported); e.g. uv venv --python 3.12 .venv
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
python3 scripts/doctor.py
python3 scripts/dry_run.py
python3 scripts/generate_sodh.py && python3 scripts/audit_sodh.py   # M1: synthetic SODH v1 (frozen fixture) + G0
python3 scripts/generate_sodh2.py && python3 scripts/run_v2_eval.py  # SODH v2 + four memory metrics (synthetic)
```

`dry_run.py` tests only a *hand-written symbolic toy baseline*. It does **not** demonstrate learned intelligence, VLA integration, RL, or benchmark improvements. No remote jobs are submitted.

## Agent handoff

1. Read `AGENTS.md`, then `agents/MASTER_IMPLEMENTATION_PROMPT.md`.
2. Implement one milestone per PR; keep baseline and data split frozen.
3. Always ship the raw log, config digest, code commit, seeds and attribution/latency diagnostics with each reported metric.
4. **No real-robot execution without a local human operator and AERI authorization.**

## Integration boundary

This is a **separate repository**. Integration with `Koooki3/LEVI`, `Physical-Intelligence/openpi`, RoboMME and hardware is proposed, not implemented. Keep existing LEVI event journal/AERI motion authority unchanged. See `docs/13_LEVI_BRIDGE.md`.

## Research integrity

- Never mix future frames, oracle simulator state, or demonstration labels into an online context policy.
- Pre-register baselines, compute budgets, paired seeds and failure criteria *before* large GPU runs.
- Report wall-clock, GPU-hours, memory, p50/p95 latency, failure modes and variance, not only success rate.
- Third-party code/dataset/model licenses must be reviewed before integration; no source-code copying until approved.

## Selected references

- ECoMEM — https://arxiv.org/abs/2610.00801
- RB-VLA — https://arxiv.org/abs/2602.20659
- Divide-and-Remember — https://arxiv.org/abs/2610.00982
- Mem4Act — https://arxiv.org/abs/2610.22323
- NativeMEM — https://arxiv.org/abs/2607.06678
- EventVLA — https://arxiv.org/abs/2606.20092
- EvoMem-VLA — https://arxiv.org/abs/2610.05418
- MOST — https://arxiv.org/abs/2611.00734
- RoboCasa-Mem — https://arxiv.org/abs/2611.06475
- Context Language Models — https://arxiv.org/abs/2609.37725
- Benchmark audit — https://arxiv.org/abs/2606.04233

No publication results exist for UC-VLA at this point.
