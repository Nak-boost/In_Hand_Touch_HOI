"""Collision semantics and stale-data protection, independent of Isaac Gym."""
from pathlib import Path
import itertools
import pytest

from hora.utils.hand_collision import collision_spec, collision_signature, require_collision_signature

ROOT = Path(__file__).resolve().parents[1]


def test_masks_exclude_only_adjacent_or_fixed_assembly_pairs():
    spec = collision_spec(ROOT/'assets/allegro/allegro_digit.urdf')
    excluded = set(map(tuple, spec['excluded_pairs']))
    masks = spec['body_masks']
    assert len(excluded) == 28
    for a, b in itertools.combinations(sorted(masks), 2):
        assert bool(masks[a] & masks[b]) == ((a, b) in excluded)
    # Different fingers and even two gels must physically collide.
    for a, b in [('link_2.0', 'link_6.0'), ('digit_index_gel', 'digit_thumb_gel'),
                 ('digit_index_housing', 'digit_middle_housing'),
                 ('base_link', 'digit_index_gel'), ('link_1.0', 'link_3.0')]:
        assert masks[a] & masks[b] == 0
    assert masks['link_3.0'] & masks['digit_index_gel']
    assert masks['digit_index_gel'] & masks['digit_index_housing']
    # A filter-zero object collides with every hand part.
    assert all(mask & 0 == 0 for mask in masks.values())


def test_old_cache_or_checkpoint_cannot_silently_resume():
    spec = collision_spec(ROOT/'assets/allegro/allegro_digit.urdf')
    signature = collision_signature(spec)
    for old in [{}, {'collision_sha256': 'disabled'}]:
        with pytest.raises(ValueError, match='collision settings mismatch'):
            require_collision_signature(old, signature, 'Cache/checkpoint')
    require_collision_signature({'collision_sha256': signature}, signature, 'Cache/checkpoint')
