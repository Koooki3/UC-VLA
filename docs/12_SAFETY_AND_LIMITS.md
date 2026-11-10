# 12 | Safety, governance and epistemic limits

## Real robot

- Real-robot motion must have an explicit authorized human at the platform, emergency stop and independent controller. The context model never owns motion authority.
- Shadow mode is mandatory before any physical motion test; all real-robot changes require an approved protocol.
- Keep task-level safety envelopes, collision detection, force/velocity limits and operator halt independent of ML context.
- No generative exploration that could damage equipment, people or lab infrastructure without human-reviewed controls.

## Data

- Evidence log immutable and checksummed. Hypotheses are labeled estimates; not rewritten as physical facts.
- Any model-generated labels cannot become ground truth through repeated summarization.
- No test split labels in training, retrieval, prompt examples, data-dependent thresholds, or memory indices.
- Licensed materials and lab captures remain access-controlled; no forced public release when prohibited.

## Model limitations

- Unlimited arbitrary histories cannot be losslessly encoded into fixed memory; dynamic external evidence only partly mitigates this.
- Memory sufficiency estimates are statistical and may miscalibrate in OOD scenarios.
- Q-value based intervention estimates may suffer unobserved confounding on real data, model bias, simulator gap and high variance.
- Async memory updates can be delayed, and eventual consistency is not safe for all control decisions; use explicit version gating.

## Red-team tests

- Timestamp corruption; future frame injection; replay attack; episode-ID collision; source hash mismatch.
- Context poisoning: false `task_complete`, fake high confidence, bogus evidence ref, malicious metadata string.
- Retrieval outage and evidence permanently lost.
- Stale rolling suffix after target swap / safety constraint update.
- Force sensor lag, dropped camera frames, inconsistent commanded vs executed actions.

## Reporting

Maintain incident record, failed seeds, aborted trials and fallback counts. Do not suppress bad trials or reclassify them as unrelated issues post hoc.
