"""Shared simulation/deployment joint-target convention (no simulator imports)."""


def action_reference(config):
    # Snapshots/checkpoints predating this field used accumulated targets.
    reference = config.get('action_reference', 'previous_target')
    if reference not in ('actual_position', 'previous_target'):
        raise ValueError('Unknown action_reference: ' + str(reference))
    return reference


def joint_targets(q, previous_targets, increment, lower, upper, reference):
    if reference == 'actual_position':
        base = q
    elif reference == 'previous_target':
        base = previous_targets
    else:
        raise ValueError('Unknown action_reference: ' + str(reference))
    targets = (base + increment).clamp(lower, upper)
    return targets, targets - base


def require_action_reference(checkpoint, config, context='Checkpoint'):
    saved, expected = action_reference(checkpoint), action_reference(config)
    if saved != expected:
        raise ValueError('%s action_reference mismatch: %s != %s. '
                         'Use the original control configuration for replay or train a new policy.'
                         % (context, saved, expected))
