"""Verify replay task signals against object-only contact loads, frame by frame."""
from pathlib import Path
import json
import numpy as np

out = Path(__file__).resolve().parent
with np.load(out/'tabletop_replay.npz') as archive:
    data = {k: archive[k] for k in archive.files}
effective = data['pair_load'] > .1
assert np.array_equal(data['task'], effective)
assert np.allclose(data['reward_contact'], .05*(effective.sum(-1) >= 2))
missing = np.zeros(effective.shape[1], dtype=np.int64)
raw_missing = missing.copy()
raw_failure = []
for step, contact in enumerate(effective):
    missing = np.where(contact.any(-1), 0, missing+1)
    assert np.array_equal(data['missing'][step], missing)
    outside = np.square(data['object'][step,:,:2]).sum(-1) > .1**2
    expected_failure = (missing >= 40) | outside
    assert np.array_equal(data['failed'][step], expected_failure)
    assert np.allclose(data['reward_failure'][step], -250*expected_failure)
    raw_missing = np.where(data['raw'][step].any(-1), 0, raw_missing+1)
    raw_failure.append((raw_missing >= 40) | outside)
assert np.array_equal(data['obs'], data['history_last'])
assert np.all(data['obs'][1:,:,64:68] <= data['raw'][:-1])
results = dict(environment_steps=int(np.prod(effective.shape[:2])),
    task_mask_matches_pair_load=True, contact_reward_matches=True,
    contact_loss_timer_matches=True, failure_and_penalty_match=True,
    removed_contact_reward_frames=int(((data['raw'].sum(-1)>=2) & (effective.sum(-1)<2)).sum()),
    failure_while_raw_counterfactual_alive_frames=int((data['failed'] & ~np.array(raw_failure)).sum()),
    raw_tactile_without_effective_object_frames=int((data['raw'].any(-1) & ~effective.any(-1)).sum()),
    note='Diagnostic replay disables auto-reset. Counts are environment frames, not independent episodes.')
assert results['removed_contact_reward_frames'] > 0
assert results['failure_while_raw_counterfactual_alive_frames'] > 0
(out/'verification.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
