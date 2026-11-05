from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
from pathlib import Path
import sys
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
import hierarchical_pair_anchor_models_v25 as base
REFERENCE_FULL = 'r12_reference_full'
LEARNED_STATIC_TEMPORAL = 'r12_learned_static_temporal'
LEARNED_STATIC_BRANCH = 'r12_learned_static_branch'
LEARNED_STATIC_BOTH = 'r12_learned_static_both'
R12_VARIANTS = (REFERENCE_FULL, LEARNED_STATIC_TEMPORAL, LEARNED_STATIC_BRANCH, LEARNED_STATIC_BOTH)
PRIMARY_CONTROLS = R12_VARIANTS[1:]
STATIC_PARAMETER_NAMES = ('static_temporal_logits', 'static_branch_logits')
ADDED_REGISTERED_PARAMETERS = base.WINDOWS * base.BRANCHES + base.BRANCHES
R12_POLICIES = {REFERENCE_FULL: {'static_temporal': False, 'static_branch': False}, LEARNED_STATIC_TEMPORAL: {'static_temporal': True, 'static_branch': False}, LEARNED_STATIC_BRANCH: {'static_temporal': False, 'static_branch': True}, LEARNED_STATIC_BOTH: {'static_temporal': True, 'static_branch': True}}
MODEL_FAMILY = 'r12_v27_trainable_global_static_selection_controls'
FULL_VARIANT = REFERENCE_FULL
STRUCTURED_VARIANTS = tuple(base.STRUCTURED_VARIANTS) + R12_VARIANTS
SINGLE_VARIANTS = tuple(base.SINGLE_VARIANTS)
VARIANTS = STRUCTURED_VARIANTS + SINGLE_VARIANTS
DETAIL_KEYS = base.DETAIL_KEYS
REGISTRY_SHA256 = base.REGISTRY_SHA256
CONTROLLER_STATE_KEYS = base.CONTROLLER_STATE_KEYS
ANCHOR_MIXING_STATE_KEYS = base.ANCHOR_MIXING_STATE_KEYS
PRIMARY_ABLATIONS = dict(base.PRIMARY_ABLATIONS)
VARIANT_POLICY = {key: dict(value) for (key, value) in base.VARIANT_POLICY.items()}
for (_variant, _policy) in R12_POLICIES.items():
    VARIANT_POLICY[_variant] = {**base.VARIANT_POLICY[base.FULL_VARIANT], 'temporal_mode': 'learned_global_static' if _policy['static_temporal'] else 'controller_dynamic_four_route', 'branch_mode': 'learned_global_static' if _policy['static_branch'] else 'adaptive_marker_only'}
GRAPH_CONTRACT = {}
parameter_count = base.parameter_count
trainable_parameter_count = base.trainable_parameter_count

class StaticSelectionModel(base.HierarchicalPairAnchorModelV25):

    def __init__(self, variant=REFERENCE_FULL, dropout=0.15):
        if variant not in R12_VARIANTS:
            raise ValueError(f'Unknown R12 variant: {variant}')
        super().__init__(base.FULL_VARIANT, dropout=dropout)
        self.r12_variant = variant
        self.static_temporal_logits = nn.Parameter(torch.zeros(base.WINDOWS, base.BRANCHES))
        self.static_branch_logits = nn.Parameter(torch.zeros(base.BRANCHES))

    def _temporal_weights(self, role_support, raw_state):
        if not R12_POLICIES[self.r12_variant]['static_temporal']:
            return super()._temporal_weights(role_support, raw_state)
        logits = self.static_temporal_logits.unsqueeze(0).expand(len(raw_state), -1, -1)
        weights = torch.softmax(logits, dim=1) * float(base.WINDOWS)
        return (weights, logits, torch.zeros_like(logits), logits)

    def _branch_weights(self, branch_marker):
        if not R12_POLICIES[self.r12_variant]['static_branch']:
            return super()._branch_weights(branch_marker)
        logits = self.static_branch_logits.unsqueeze(0).expand(len(branch_marker), -1)
        return (torch.softmax(logits, dim=-1), torch.zeros_like(logits), logits)

def make_model(variant=REFERENCE_FULL, dropout=0.15):
    if variant in R12_VARIANTS:
        return StaticSelectionModel(variant, dropout)
    return base.make_model(variant, dropout)

def capacity_record():
    instance = make_model()
    original = base.make_model()
    return {'original_registered_parameters': parameter_count(original), 'r12_registered_parameters': parameter_count(instance), 'added_registered_parameters': ADDED_REGISTERED_PARAMETERS, 'additional_static_parameters_used': {REFERENCE_FULL: 0, LEARNED_STATIC_TEMPORAL: 32, LEARNED_STATIC_BRANCH: 4, LEARNED_STATIC_BOTH: 36}, 'static_softmax_free_dimensions': {REFERENCE_FULL: 0, LEARNED_STATIC_TEMPORAL: 28, LEARNED_STATIC_BRANCH: 3, LEARNED_STATIC_BOTH: 31}}

def self_test():
    with torch.random.fork_rng(devices=[]):
        states = []
        for variant in R12_VARIANTS:
            torch.manual_seed(2026102312)
            instance = make_model(variant)
            states.append(instance.state_dict())
        assert all((set(state) == set(states[0]) for state in states))
        assert all((torch.equal(value, state[key]) for state in states[1:] for (key, value) in states[0].items()))
        assert capacity_record()['added_registered_parameters'] == 36
    return {'status': 'PASS', 'variants': list(R12_VARIANTS), 'equal_initial_state': True}
