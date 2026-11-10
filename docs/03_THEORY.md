# 03 | Theory: Decision-sufficient compression under latency and evidence constraints

## 1. Formal model

Let a partially observed control process have latent state `s_t`, observation `o_t`, task `g`, control action `a_t`, verified feedback `y_t`, and causal history `h_t=(g,o_0,a_0^exec,y_1,o_1,...,o_t)`. The full Bayesian belief `b_t=P(s_t|h_t)` is sufficient when the model is known, but exact storage/inference is impractical. Our model produces `z_t=F_phi(z_{t-1},o_t,a_{t-1}^exec,y_t,g,u_t,r_t)` where `u_t` is a memory operation, `r_t` is a causal evidence retrieval, with bounded fast memory, optional immutable episodic store, and a latency budget.

Policy: `a_t ~ pi_theta(. | o_t,g,z_t)` or action chunk `A_{t:t+H-1}` with rolling updates. Memory policy: `u_t ~ mu_phi(.|o_t,g,z_{t-1},B_t)`; read and write both consume compute.

## 2. Goal: a constrained POMDP, not perfect reconstruction

Maximize expected discounted closed-loop reward with real measured compute/latency budgets:

```
max_{theta,phi} E[sum_t gamma^t r(s_t,a_t)]
s.t. E[C_fast(z_t)] <= B_memory;
     E[C_update+C_retrieve+C_policy] <= B_compute;
     P(L_control > deadline) <= delta_latency;
     P(unsafe_action) <= delta_safety.
```

Lagrangian approximations for training are proxies; a penalty does not replace a hard runtime deadline or safety interlock. Memory storage cost includes external evidence store and its I/O when reporting total system cost.

## 3. Decision-value distortion (most important theorem-shaped hypothesis)

For candidate action set A(h_t) (same legal actions under full and compressed views), define `Q*(h_t,a)` as the true full-history closed-loop action value, `Qhat(o_t,g,z_t,a)` as estimate from compressed state:

```
d_Q(h_t,z_t) = sup_{a in A(h_t)} | Q*(h_t,a) - Qhat(o_t,g,z_t,a) |.
```

**Conditional greedy regret lemma.** If for a fixed state, the bound `d_Q <= epsilon` holds for all candidate actions, and `a_hat=argmax_a Qhat(o_t,g,z_t,a)` while `a*=argmax_a Q*(h_t,a)`, then:

```
Q*(h_t,a*) - Q*(h_t,a_hat)
<= [Q*(h_t,a*)-Qhat(a*)] + [Qhat(a_hat)-Q*(h_t,a_hat)] <= 2 epsilon.
```

The inequality relies on `Qhat(a_hat) >= Qhat(a*)`; this is *one-step value regret relative to the true Q* from the same history*. It is not a verified bound on the neural policy's full episode regret. Extending to the rollout requires assumptions on errors under policy-induced states, action-set coverage and stability; distribution shift and Q estimator error can invalidate it. Never write the theorem as a success-rate guarantee.

**Key distinction from D&R:** the memory is trained to preserve *counterfactual future action values*, not only conditional information about the present expert action `I(A_t^expert; M_t | O_t)`. Whether this is empirically useful is to be tested.

## 4. Practical estimators without privileged online leakage

`Q*` is inaccessible on real robots. In resettable simulators construct paired branches with identical true state, same observation prefix and controlled random seeds, differing by memory interventions. Use episodic return differences as a noisy empirical estimator of a memory item's future utility; use FQE / distributional critic ensembles as secondary estimators. Offline historical trajectories alone do **not** identify counterfactual returns without overlap and policy/value assumptions. We will explicitly report that limitation.

For an evidence item `e_i`, approximate `Delta_i = E[R | do(keep e_i)] - E[R | do(remove e_i)]` under a fixed policy and rollout protocol. **Ablation changes policy conditioning**, not world physics. Counterfactual memory intervention is valid only when simulator states and observation cutoffs are held identical; item interactions require group/conditional Shapley-like sampling rather than naive additive credit.

## 5. Recoverability and value of information

For retrieving evidence/performing active observation `q`, define:

```
VOI(q) = E_{y|q}[max_a E[Q(a)|z,y]] - max_a E[Q(a)|z] - C(q).
```

If the posterior is wrong, or a key event was never recorded, no algorithm can guarantee recovery. Learn `p_suff` and calibrate on held-out OOD data; disallow confident execution when confidence is low and high-risk action is contemplated. C(q) includes latency, robot travel and physical risk, not just tokens.

## 6. Decision-aware rate–distortion

`min E[d_Q(h,z)] + beta C_representation(z) + eta C_compute(z)`, conditional on goal and current observation. This is an objective, not a claim that a universally finite sufficient statistic exists. Random histories requiring arbitrary exact recall impose information-theoretic lower bounds; adapt budget, retrieve from external storage or acknowledge inability.

## 7. Partial observability and belief calibration

A fixed vector z_t generally cannot represent the exact belief over arbitrary environments. Test beliefs on event/state probes and posterior calibration under occlusion; if identical observations induce diverse future actions, use mixture/state distribution rather than point estimates. Add provenance and unknown-state representations so the model can abstain.

## 8. Falsification

If Q-guided compression under equal cost gives no statistically robust benefit over action-MI compression, or learned Q is too inaccurate under distribution shift, this proposed theoretical objective has not earned its extra complexity. Do not bury this finding behind larger architecture gains.
