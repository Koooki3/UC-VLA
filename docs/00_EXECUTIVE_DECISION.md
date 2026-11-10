# 00 | 审查后的研究决策

## Objective

以顶会/顶刊标准开发一个**统一决策上下文模型**（UC-VLA）。在历史依赖的长程机器人任务中，使策略在相同或更低资源预算下更可靠地决策和恢复，而非通过简单增长窗口与新增大型 Planner 换取成功率。目标刊会：CoRL / ICLR / NeurIPS / T-RO（需要根据实际结果与投稿期限择一；没有成功保证）。

## 同一性定义（research interpretation）

同一物理事件在观测、历史证据、任务状态、动作规划和执行验证之间有**一致且可追溯的时空身份与语义**。唯一可信的过去是时间戳/原始哈希绑定的已观测或已执行事件；模型对过去的解释可更正。新的动作必须基于某个已提交的上下文版本。

## Central hypothesis

相对简单的固定压缩、Action-Information 选择和 Task-State Memory，**以可验证的未来闭环决策损失为训练信号、且能按成本主动恢复的记忆策略**，能够在跨历史长度/对象数/遮挡分布的测试中改善 *success–cost–latency Pareto frontier*。

必须同时满足三个证据条件：

1. **Causality**：相同当前观测、不同历史导致正确动作不同；替换正确/错误 Memory 能可预测地改变结果，杜绝 shortcut。
2. **Efficiency**：包括 Memory Writer、Reader、Retrieval、计划失效和 Policy 的端到端成本，不能只报告 Token 数。
3. **Generalization**：未见过的 episode length、event count、对象映射、干扰强度下仍改善；不只追求训练分布内性能。

## Three principal contributions (proposed, not proven)

- **Theory:** Action-Conditioned / Future Decision-Sufficient Compression，给出条件化 value distortion 和 regret 上界，并明确不是任意任务的固定维充分统计量。
- **Learning:** 决策损失驱动 Memory Policy + Sufficiency/Value-of-Information 管理，用可重置仿真状态的干预实验构建信号，避免把 expert current-action loss 当长期状态充分性。
- **Systems:** Evidence-ledger/Versioned Context/Plan-Validity 协定，与实时 rolling action chunk 解耦；在相同硬件与吞吐预算下验证。

如果缺少任一项，不应夸大为 unified architecture；可缩减为清晰的单一贡献论文。

## Earliest killer experiment

冻结一个可运行的 VLA 与数据源。构造 100+ 对（数量是初始工程目标，不是统计功效证明）**同当前观测、不同历史、不同正确动作**的测试案例；实现 Memoryless、Full History、Random/Uniform, D&R-like and Decision-aware 两种压缩器；用等 Token 与等 wall-clock 双预算比较。检查记忆置换、因果解释与成功率。有明显优势且没有推理延迟退化，才进入 RL。

## Go / no-go / pivot

- **G0**（数据有效）：memoryless 下历史交换不能改变动作，且任务不随历史变化 → 该任务无诊断力，重做数据。
- **G1**（架构有效）：Memory Reader 无法可靠改变动作且不胜简单固定窗口 → 停止复杂 RL，改进 Reader/训练数据。
- **G2**（新贡献有效）：Decision-aware 相对 D&R/最强可复现 baseline 在等预算差异不稳定或无效 → 不宣称新压缩策略优势；考虑诊断性 benchmark 或反例研究。
- **G3**（实际价值）：端到端 p95 延迟超出硬件允许值、或出现安全违规 → 不进入真机；改为异步更新或限制模型规模。
- **G4**（发表就绪）：无跨方法、跨种子、OOD、真实成本、失败分析 → 不投稿为强实证论文。

此 Blueprint 的结论是**继续做，但收窄新意并强化可证伪性**。见 `08_ADVERSARIAL_REVIEWS.md`。
