# PPO action-mean boundary penalty (2026-10-13)

The corrected per-sample penalty is
`sum(relu(mu - b)^2 + relu(-mu - b)^2)` over action dimensions.
It has zero loss and gradient inside `[-b, b]` and treats positive and
negative means equally. PPO still averages this loss across the minibatch.
This changes an optimization regularizer, not the environment reward.

The previous upstream expression was equivalent to `sum((mu - 1.1)^2)`.
The three completed `teacher_*_z_self_collision_v1` runs used that old
expression. Their saved configurations, source snapshots and weights are
historical evidence and have not been rewritten.

Configuration:

- `train.ppo.bounds_loss_soft_bound=1.1` (new configurable default).
- `train.ppo.bounds_loss_coef=0.0001` (existing coefficient retained).
- Executed actions continue to be clipped to `[-1, 1]`.
- Setting the coefficient to zero disables the penalty; the disabled
  branch now supplies a zero tensor so reduction and logging remain valid.

Keep `b=1.1` for the first corrected baseline to isolate the formula change.
It permits means up to 10% beyond the execution range without a penalty;
this is a soft margin, not an experimentally established optimum.
`b=1.0` is a reasonable separate comparison when earlier discouragement
of mean saturation is desired. Neither value constrains Gaussian samples
directly or guarantees that accumulated joint targets avoid their limits.
Compare sustained rotation, mean/sample clipping fractions, joint-target
limit fractions and retention, rather than return alone.

Validation: `scripts/python_manip.sh -m pytest -q tests` passed all 11 cases,
including four regression cases checking zero interior loss/gradient,
symmetry and inward gradients for both `b=1.0` and `b=1.1`.
No new training or policy-performance claim accompanies this correction.
