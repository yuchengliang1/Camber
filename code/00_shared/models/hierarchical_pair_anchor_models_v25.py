from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(next((p for p in Path(__file__).resolve().parents if (p / 'project_paths.py').is_file()))))
from project_paths import PACKAGE as _PACKAGE, RESULTS as _RESULTS, resolve_path
import math
from typing import Final
import torch
from torch import nn
from torch.nn import functional as F
from feature_coefficients import ANCHOR_ROUTE_ANCHOR, ANCHOR_ROUTE_STRUCTURED, BANDS, BRANCHES, BRANCH_EEG, BRANCH_FUSION, BRANCH_MEG, BRANCH_PAIR, CHANNELS, CLASSES, CONTROLLER_MODAL_ROUTE, CONTROLLER_PAIR_EXPERT, CONTROLLER_PATH_DIRECTION, EXPERT_DIFFERENCE, EXPERT_PRODUCT, MODALITY_EEG, MODALITY_MEG, PATH_EEG_TO_MEG, PATH_MEG_TO_EEG, REGISTRY_SHA256, WINDOWS, path_endpoint_for_modality, registry_arrays
EPS: Final = 1e-08
SAMPLE_RATE_HZ: Final = 250.0
SAMPLES_PER_WINDOW: Final = 125
TOKEN_DIM: Final = 16
PAIR_TOKEN_DIM: Final = 8
BAND_LIMITS_HZ: Final = ((8.0, 13.0), (18.0, 30.0))
BRANCH_PUBLIC_SCALE: Final = 2.25
BRANCH_RESIDUAL_SCALE: Final = 0.12
TEMPORAL_PUBLIC_SCALE: Final = 1.75
TEMPORAL_RESIDUAL_SCALE: Final = 0.12
EXPERT_RESIDUAL_SCALE: Final = 0.15
MIX_PUBLIC_SCALE: Final = 2.25
MIX_RESIDUAL_SCALE: Final = 0.12
MIX_MIN: Final = 0.02
MIX_RANGE: Final = 0.96
CALIBRATION_MIN_SCALE: Final = 0.04
FULL_VARIANT: Final = 'hierarchical_pair_anchor_v25_full'
NO_BIDIRECTIONAL_PATH_CONTROLLER: Final = 'hierarchical_pair_anchor_v25_no_bidirectional_path_controller'
MATCHED_CONCAT: Final = 'hierarchical_pair_anchor_v25_matched_concat'
UNIFORM_TEMPORAL: Final = 'hierarchical_pair_anchor_v25_uniform_temporal'
FIXED_BRANCH_GATE: Final = 'hierarchical_pair_anchor_v25_fixed_branch_gate'
FIXED_ANCHOR_MIXING: Final = 'hierarchical_pair_anchor_v25_fixed_anchor_mixing'
EEG_ONLY: Final = 'hierarchical_pair_anchor_v25_eeg_only'
MEG_ONLY: Final = 'hierarchical_pair_anchor_v25_meg_only'
STRUCTURED_VARIANTS: Final = (FULL_VARIANT, NO_BIDIRECTIONAL_PATH_CONTROLLER, MATCHED_CONCAT, UNIFORM_TEMPORAL, FIXED_BRANCH_GATE, FIXED_ANCHOR_MIXING)
SINGLE_VARIANTS: Final = (EEG_ONLY, MEG_ONLY)
VARIANTS: Final = STRUCTURED_VARIANTS + SINGLE_VARIANTS
MODEL_FAMILY: Final = 'v25_coordinate_isolated_bidirectional_dual_expert_four_route_four_head_anchor'
PRIMARY_ABLATIONS: Final = {'bidirectional_path_controller': NO_BIDIRECTIONAL_PATH_CONTROLLER, 'dual_expert_pair_interaction': MATCHED_CONCAT, 'dynamic_four_route_temporal_pooling': UNIFORM_TEMPORAL, 'adaptive_four_head_aggregation': FIXED_BRANCH_GATE, 'adaptive_anchor_probability_mixing': FIXED_ANCHOR_MIXING}
VARIANT_POLICY: Final = {FULL_VARIANT: {'path_enabled': True, 'controller_enabled': True, 'interaction_mode': 'product_abs_difference', 'temporal_mode': 'controller_dynamic_four_route', 'branch_mode': 'adaptive_marker_only', 'anchor_mixing_mode': 'adaptive_anchor_route_only'}, NO_BIDIRECTIONAL_PATH_CONTROLLER: {'path_enabled': False, 'controller_enabled': False, 'interaction_mode': 'product_abs_difference', 'temporal_mode': 'exact_uniform_without_controller', 'branch_mode': 'adaptive_marker_only', 'anchor_mixing_mode': 'adaptive_anchor_route_only'}, MATCHED_CONCAT: {'path_enabled': True, 'controller_enabled': True, 'interaction_mode': 'matched_eeg_meg_concat', 'temporal_mode': 'controller_dynamic_four_route', 'branch_mode': 'adaptive_marker_only', 'anchor_mixing_mode': 'adaptive_anchor_route_only'}, UNIFORM_TEMPORAL: {'path_enabled': True, 'controller_enabled': True, 'interaction_mode': 'product_abs_difference', 'temporal_mode': 'exact_uniform', 'branch_mode': 'adaptive_marker_only', 'anchor_mixing_mode': 'adaptive_anchor_route_only'}, FIXED_BRANCH_GATE: {'path_enabled': True, 'controller_enabled': True, 'interaction_mode': 'product_abs_difference', 'temporal_mode': 'controller_dynamic_four_route', 'branch_mode': 'exact_quarters', 'anchor_mixing_mode': 'adaptive_anchor_route_only'}, FIXED_ANCHOR_MIXING: {'path_enabled': True, 'controller_enabled': True, 'interaction_mode': 'product_abs_difference', 'temporal_mode': 'controller_dynamic_four_route', 'branch_mode': 'adaptive_marker_only', 'anchor_mixing_mode': 'fixed_outer_train_lambda'}}
CONTROLLER_STATE_KEYS: Final = ('controller_centers', 'controller_scales', 'controller_calibration_fitted')
ANCHOR_MIXING_STATE_KEYS: Final = ('anchor_route_center', 'anchor_route_scale', 'anchor_fixed_lambda', 'anchor_mixing_calibration_fitted')
DETAIL_KEYS: Final = ('direct_eeg_token', 'direct_meg_token', 'pair_token', 'path_eeg_to_meg_residual', 'path_meg_to_eeg_residual', 'fusion_input', 'fusion_token', 'exchange_policy', 'controller_state', 'controller_membership', 'controller_role_support', 'interaction_weights', 'temporal_logits', 'temporal_public_logits', 'temporal_learned_residual', 'temporal_weights', 'branch_marker', 'branch_weights', 'branch_public_logits', 'branch_learned_residual', 'branch_logits', 'anchor_task_feature', 'anchor_route_feature', 'anchor_public_cue', 'anchor_pretraining_weight', 'anchor_fixed_lambda', 'anchor_probability', 'structured_probability', 'mix_weight')
GRAPH_CONTRACT: Final = {}

def parameter_count(model: nn.Module) -> int:
    return sum((parameter.numel() for parameter in model.parameters()))

def trainable_parameter_count(model: nn.Module) -> int:
    return sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))

class _CoordinatePowerBank(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        arrays = registry_arrays(copy=True)
        self.register_buffer('direct_task', torch.as_tensor(arrays['direct_task'], dtype=torch.float32), persistent=True)
        self.register_buffer('pair_task', torch.as_tensor(arrays['pair_task'], dtype=torch.float32), persistent=True)
        self.register_buffer('path_task', torch.as_tensor(arrays['path_task'], dtype=torch.float32), persistent=True)
        self.register_buffer('anchor_task', torch.as_tensor(arrays['anchor_task'], dtype=torch.float32), persistent=True)
        self.register_buffer('controller', torch.as_tensor(arrays['controller'], dtype=torch.float32), persistent=True)
        self.register_buffer('branch', torch.as_tensor(arrays['branch'], dtype=torch.float32), persistent=True)
        self.register_buffer('anchor_route', torch.as_tensor(arrays['anchor_route'], dtype=torch.float32), persistent=True)
        frequencies = torch.fft.rfftfreq(WINDOWS * SAMPLES_PER_WINDOW, d=1.0 / SAMPLE_RATE_HZ)
        masks = torch.stack([((frequencies >= float(low)) & (frequencies <= float(high))).to(dtype=torch.float32) for (low, high) in BAND_LIMITS_HZ], dim=0)
        if tuple(masks.shape) != (BANDS, WINDOWS * SAMPLES_PER_WINDOW // 2 + 1):
            raise AssertionError(tuple(masks.shape))
        self.register_buffer('band_fft_masks', masks, persistent=True)

    @staticmethod
    def _validate(values: torch.Tensor) -> None:
        if values.ndim != 3 or values.shape[1] != CHANNELS or values.shape[2] != WINDOWS * SAMPLES_PER_WINDOW:
            raise ValueError(f'Expected V25 public signal (N,{CHANNELS},{WINDOWS * SAMPLES_PER_WINDOW}), got {tuple(values.shape)}')

    @staticmethod
    def _repeat_across_bands(maps: torch.Tensor) -> torch.Tensor:
        return maps.unsqueeze(0).expand(BANDS, -1, -1)

    def _band_limited_values(self, values: torch.Tensor) -> torch.Tensor:
        self._validate(values)
        spectrum = torch.fft.rfft(values, dim=-1)
        masks = self.band_fft_masks.to(dtype=spectrum.real.dtype, device=values.device)
        filtered = torch.fft.irfft(spectrum.unsqueeze(1) * masks.view(1, BANDS, 1, -1), n=values.shape[-1], dim=-1)
        return filtered

    def _window_log_power(self, values: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        self._validate(values)
        if maps.ndim != 3 or maps.shape[0] != BANDS or maps.shape[-1] != CHANNELS:
            raise ValueError(f'Expected (B,K,C) V25 coordinate maps, got {tuple(maps.shape)}')
        directions = F.normalize(maps.to(dtype=values.dtype, device=values.device), dim=-1, eps=EPS)
        projected = torch.einsum('nbct,bkc->nbkt', self._band_limited_values(values), directions)
        shaped = projected.reshape(len(values), BANDS, maps.shape[1], WINDOWS, SAMPLES_PER_WINDOW)
        power = shaped.var(dim=-1, unbiased=False).clamp_min(1e-06).log()
        return power.permute(0, 3, 1, 2).reshape(len(values), WINDOWS, -1)

    def direct_features(self, values: torch.Tensor, modality: int) -> torch.Tensor:
        return self._window_log_power(values, self.direct_task[int(modality)])

    def pair_features(self, values: torch.Tensor, modality: int) -> torch.Tensor:
        maps = self.pair_task[int(modality)].permute(1, 0, 2)
        return self._window_log_power(values, maps)

    def path_features(self, values: torch.Tensor, direction: int, modality: int) -> torch.Tensor:
        endpoint = path_endpoint_for_modality(int(direction), int(modality))
        maps = self.path_task[int(direction), int(endpoint)].unsqueeze(1)
        return self._window_log_power(values, maps)

    def anchor_features(self, values: torch.Tensor, modality: int) -> torch.Tensor:
        return self._window_log_power(values, self._repeat_across_bands(self.anchor_task[int(modality)]))

    def controller_features(self, values: torch.Tensor, modality: int) -> torch.Tensor:
        return self._window_log_power(values, self._repeat_across_bands(self.controller[int(modality)]))

    def branch_features(self, values: torch.Tensor, modality: int) -> torch.Tensor:
        return self._window_log_power(values, self._repeat_across_bands(self.branch[int(modality)]))

    def anchor_route_features(self, values: torch.Tensor, modality: int) -> torch.Tensor:
        return self._window_log_power(values, self._repeat_across_bands(self.anchor_route[int(modality)]))

    @staticmethod
    def reduce_bands(features: torch.Tensor, coordinates: int) -> torch.Tensor:
        if features.shape[-1] != BANDS * int(coordinates):
            raise ValueError((tuple(features.shape), coordinates))
        return features.reshape(*features.shape[:-1], BANDS, int(coordinates)).mean(dim=-2)

    def controller_state(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        eeg_feature = self.reduce_bands(self.controller_features(eeg, MODALITY_EEG), 3)
        meg_feature = self.reduce_bands(self.controller_features(meg, MODALITY_MEG), 3)
        return torch.stack((meg_feature[..., CONTROLLER_PAIR_EXPERT], eeg_feature[..., CONTROLLER_MODAL_ROUTE], 0.5 * (eeg_feature[..., CONTROLLER_PATH_DIRECTION] + meg_feature[..., CONTROLLER_PATH_DIRECTION])), dim=-1)

    def branch_marker(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        eeg_feature = self.reduce_bands(self.branch_features(eeg, MODALITY_EEG), BRANCHES)
        meg_feature = self.reduce_bands(self.branch_features(meg, MODALITY_MEG), BRANCHES)
        return 0.5 * (eeg_feature + meg_feature)

    def anchor_route_marker(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        eeg_feature = self.reduce_bands(self.anchor_route_features(eeg, MODALITY_EEG), 2)
        meg_feature = self.reduce_bands(self.anchor_route_features(meg, MODALITY_MEG), 2)
        return 0.5 * (eeg_feature + meg_feature)

class _Head(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 12), nn.Tanh(), nn.Dropout(dropout), nn.Linear(12, CLASSES))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)

class _AnchorClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 16), nn.Tanh(), nn.Linear(16, CLASSES))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)

class _SingleCoordinateModelV25(nn.Module):
    requires_controller_calibration: Final = False
    requires_anchor_mixing_calibration: Final = False
    requires_anchor_pretraining: Final = False

    def __init__(self, modality: int) -> None:
        super().__init__()
        self.modality = int(modality)
        self.coordinates = _CoordinatePowerBank()
        self.classifier = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, CLASSES))

    def forward_with_details(self, eeg: torch.Tensor, meg: torch.Tensor) -> dict[str, torch.Tensor]:
        source = eeg if self.modality == MODALITY_EEG else meg
        token = self.coordinates.direct_features(source, self.modality)
        probability = torch.softmax(self.classifier(token.mean(dim=1)), dim=-1)
        branch = source.new_zeros((len(source), BRANCHES))
        branch[:, BRANCH_EEG if self.modality == MODALITY_EEG else BRANCH_MEG] = 1.0
        return {'log_probability': probability.clamp_min(EPS).log(), 'local_weights': source.new_ones((len(source), WINDOWS, BRANCHES)), 'branch_weights': branch, 'mix_weight': source.new_ones((len(source), 1)), 'structured_probability': probability, 'anchor_probability': probability}

    def forward(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(eeg, meg)['log_probability']

class HierarchicalPairAnchorModelV25(nn.Module):
    requires_controller_calibration: Final = True
    requires_anchor_mixing_calibration: Final = True
    requires_anchor_pretraining: Final = True

    def __init__(self, variant: str=FULL_VARIANT, dropout: float=0.2) -> None:
        super().__init__()
        if variant not in STRUCTURED_VARIANTS:
            raise ValueError(f'Unknown V25 structured variant: {variant}')
        self.variant = str(variant)
        self.coordinates = _CoordinatePowerBank()
        self.eeg_direct_encoder = nn.Sequential(nn.Linear(4, TOKEN_DIM), nn.Tanh(), nn.Dropout(dropout))
        self.meg_direct_encoder = nn.Sequential(nn.Linear(4, TOKEN_DIM), nn.Tanh(), nn.Dropout(dropout))
        self.eeg_pair_encoder = nn.Sequential(nn.Linear(4, PAIR_TOKEN_DIM), nn.Tanh())
        self.meg_pair_encoder = nn.Sequential(nn.Linear(4, PAIR_TOKEN_DIM), nn.Tanh())
        self.product_expert = nn.Sequential(nn.Linear(PAIR_TOKEN_DIM, PAIR_TOKEN_DIM), nn.Tanh())
        self.difference_expert = nn.Sequential(nn.Linear(PAIR_TOKEN_DIM, PAIR_TOKEN_DIM), nn.Tanh())
        self.pair_projection = nn.Sequential(nn.Linear(2 * PAIR_TOKEN_DIM, TOKEN_DIM), nn.Tanh(), nn.Dropout(dropout))
        self.expert_selector_residual = nn.Linear(3, 2)
        self.path_eeg_to_meg = nn.Sequential(nn.Linear(8, TOKEN_DIM), nn.Tanh(), nn.Dropout(dropout))
        self.path_meg_to_eeg = nn.Sequential(nn.Linear(8, TOKEN_DIM), nn.Tanh(), nn.Dropout(dropout))
        self.fusion_projection = nn.Sequential(nn.Linear(2 * TOKEN_DIM, TOKEN_DIM), nn.Tanh(), nn.Dropout(dropout))
        self.temporal_controller_residual = nn.Linear(3, BRANCHES)
        self.branch_marker_residual = nn.Linear(BRANCHES, BRANCHES)
        self.eeg_head = _Head(TOKEN_DIM, dropout)
        self.meg_head = _Head(TOKEN_DIM, dropout)
        self.pair_head = _Head(TOKEN_DIM, dropout)
        self.fusion_head = _Head(TOKEN_DIM, dropout)
        self.anchor = _AnchorClassifier()
        self.anchor_mix_residual = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        self._anchor_frozen = False
        self.register_buffer('controller_centers', torch.zeros(3, 3), persistent=True)
        self.register_buffer('controller_scales', torch.ones(3), persistent=True)
        self.register_buffer('controller_calibration_fitted', torch.tensor(False), persistent=True)
        self.register_buffer('anchor_route_center', torch.tensor(0.0), persistent=True)
        self.register_buffer('anchor_route_scale', torch.tensor(1.0), persistent=True)
        self.register_buffer('anchor_fixed_lambda', torch.tensor(0.5), persistent=True)
        self.register_buffer('anchor_mixing_calibration_fitted', torch.tensor(False), persistent=True)
        self._initialize_declared_residuals()

    def _initialize_declared_residuals(self) -> None:
        with torch.no_grad():
            for module in (self.expert_selector_residual, self.temporal_controller_residual, self.branch_marker_residual):
                module.weight.zero_()
                module.bias.zero_()
            for parameter in self.anchor_mix_residual.parameters():
                parameter.zero_()

    def train(self, mode: bool=True) -> 'HierarchicalPairAnchorModelV25':
        super().train(mode)
        if self._anchor_frozen:
            self.anchor.eval()
        return self

    @property
    def policy(self) -> dict[str, object]:
        return VARIANT_POLICY[self.variant]

    @staticmethod
    def _weighted_mean(tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (tokens * weights.unsqueeze(-1)).sum(dim=1) / float(WINDOWS)

    @staticmethod
    def _within_feature_z(value: torch.Tensor) -> torch.Tensor:
        return (value - value.mean(dim=-1, keepdim=True)) / value.var(dim=-1, unbiased=False, keepdim=True).add(EPS).sqrt()

    @staticmethod
    def _centers_and_scales(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = state.reshape(-1, state.shape[-1])
        centers = torch.quantile(flattened, torch.tensor((0.2, 0.5, 0.8), device=state.device, dtype=state.dtype), dim=0).transpose(0, 1)
        scales = (torch.quantile(flattened, 0.75, dim=0) - torch.quantile(flattened, 0.25, dim=0)).abs()
        return (centers, scales.clamp_min(CALIBRATION_MIN_SCALE))

    def _raw_controller_state(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        return self.coordinates.controller_state(eeg, meg)

    def fit_controller_calibration(self, eeg: torch.Tensor, meg: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self._raw_controller_state(eeg, meg).detach()
        (centers, scales) = self._centers_and_scales(state)
        self.controller_centers.copy_(centers)
        self.controller_scales.copy_(scales)
        self.controller_calibration_fitted.fill_(True)
        return self.controller_calibration()

    def load_controller_calibration(self, centers: torch.Tensor, scales: torch.Tensor) -> None:
        if tuple(centers.shape) != (3, 3) or tuple(scales.shape) != (3,):
            raise ValueError((tuple(centers.shape), tuple(scales.shape)))
        self.controller_centers.copy_(centers.to(self.controller_centers))
        self.controller_scales.copy_(scales.to(self.controller_scales).clamp_min(CALIBRATION_MIN_SCALE))
        self.controller_calibration_fitted.fill_(True)

    def controller_calibration(self) -> dict[str, torch.Tensor]:
        if not bool(self.controller_calibration_fitted.item()):
            raise RuntimeError('V25 controller calibration is required before inference')
        return {'centers': self.controller_centers.detach().clone(), 'spreads': self.controller_scales.detach().clone()}

    def _controller_membership(self, raw_state: torch.Tensor) -> torch.Tensor:
        if not bool(self.controller_calibration_fitted.item()):
            raise RuntimeError('V25 controller calibration is required before inference')
        if not bool(self.policy['controller_enabled']):
            return torch.full((*raw_state.shape, 3), 1.0 / 3.0, device=raw_state.device, dtype=raw_state.dtype)
        centers = self.controller_centers.to(raw_state).view(1, 1, 3, 3)
        scales = self.controller_scales.to(raw_state).view(1, 1, 3, 1).clamp_min(CALIBRATION_MIN_SCALE)
        score = -0.5 * ((raw_state.unsqueeze(-1) - centers) / scales).square()
        return torch.softmax(score, dim=-1)

    @staticmethod
    def _role_support(membership: torch.Tensor) -> torch.Tensor:
        pair = membership[..., CONTROLLER_PAIR_EXPERT, 0] + membership[..., CONTROLLER_PAIR_EXPERT, 2]
        eeg = membership[..., CONTROLLER_MODAL_ROUTE, 0]
        meg = membership[..., CONTROLLER_MODAL_ROUTE, 2]
        fusion = membership[..., CONTROLLER_PATH_DIRECTION, 0] + membership[..., CONTROLLER_PATH_DIRECTION, 2]
        return torch.stack((eeg, meg, pair, fusion), dim=-1)

    def _route_cue(self, eeg: torch.Tensor, meg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        marker = self.coordinates.anchor_route_marker(eeg, meg)
        cue = marker[..., ANCHOR_ROUTE_STRUCTURED].mean(dim=1) - marker[..., ANCHOR_ROUTE_ANCHOR].mean(dim=1)
        return (cue, marker)

    def fit_anchor_mixing_calibration(self, eeg: torch.Tensor, meg: torch.Tensor) -> dict[str, torch.Tensor]:
        (cue, _) = self._route_cue(eeg, meg)
        center = cue.detach().mean()
        scale = cue.detach().std(unbiased=False).clamp_min(CALIBRATION_MIN_SCALE)
        public_lambda = self._public_structured_weight(cue, center=center, scale=scale)
        self.anchor_route_center.copy_(center)
        self.anchor_route_scale.copy_(scale)
        self.anchor_fixed_lambda.copy_(public_lambda.detach().mean())
        self.anchor_mixing_calibration_fitted.fill_(True)
        return self.anchor_mixing_calibration()

    def load_anchor_mixing_calibration(self, center: torch.Tensor, scale: torch.Tensor, fixed_lambda: torch.Tensor) -> None:
        self.anchor_route_center.copy_(center.to(self.anchor_route_center))
        self.anchor_route_scale.copy_(scale.to(self.anchor_route_scale).clamp_min(CALIBRATION_MIN_SCALE))
        self.anchor_fixed_lambda.copy_(fixed_lambda.to(self.anchor_fixed_lambda).clamp(MIX_MIN, MIX_MIN + MIX_RANGE))
        self.anchor_mixing_calibration_fitted.fill_(True)

    def set_fixed_anchor_lambda(self, value: float | torch.Tensor) -> None:
        value_tensor = torch.as_tensor(value, dtype=self.anchor_fixed_lambda.dtype, device=self.anchor_fixed_lambda.device)
        if value_tensor.numel() != 1:
            raise ValueError('V25 fixed Anchor lambda must be scalar')
        self.anchor_fixed_lambda.copy_(value_tensor.reshape(()).clamp(MIX_MIN, MIX_MIN + MIX_RANGE))

    def anchor_mixing_calibration(self) -> dict[str, torch.Tensor]:
        if not bool(self.anchor_mixing_calibration_fitted.item()):
            raise RuntimeError('V25 Anchor-route calibration is required before inference')
        return {'center': self.anchor_route_center.detach().clone(), 'scale': self.anchor_route_scale.detach().clone(), 'fixed_structured_mean': self.anchor_fixed_lambda.detach().clone()}

    def _public_structured_weight(self, cue: torch.Tensor, *, center: torch.Tensor | None=None, scale: torch.Tensor | None=None) -> torch.Tensor:
        use_center = self.anchor_route_center if center is None else center
        use_scale = self.anchor_route_scale if scale is None else scale
        z = (cue - use_center.to(cue)) / use_scale.to(cue).clamp_min(CALIBRATION_MIN_SCALE)
        return (MIX_MIN + MIX_RANGE * torch.sigmoid(MIX_PUBLIC_SCALE * z)).unsqueeze(-1)

    def public_anchor_weight(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        (cue, _) = self._route_cue(eeg, meg)
        return 1.0 - self._public_structured_weight(cue)

    def anchor_pretraining_weight(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        return self.public_anchor_weight(eeg, meg)

    def _interaction_weights(self, membership: torch.Tensor, raw_state: torch.Tensor) -> torch.Tensor:
        if not bool(self.policy['controller_enabled']):
            return torch.full((*raw_state.shape[:2], 2), 0.5, dtype=raw_state.dtype, device=raw_state.device)
        public = torch.stack((membership[..., CONTROLLER_PAIR_EXPERT, 2], membership[..., CONTROLLER_PAIR_EXPERT, 0]), dim=-1).clamp_min(EPS).log()
        residual = EXPERT_RESIDUAL_SCALE * torch.tanh(self.expert_selector_residual(raw_state))
        return torch.softmax(public + residual, dim=-1)

    def _pair_token(self, eeg_pair: torch.Tensor, meg_pair: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        a = self.eeg_pair_encoder(eeg_pair)
        b = self.meg_pair_encoder(meg_pair)
        if self.policy['interaction_mode'] == 'product_abs_difference':
            left = self.product_expert(a * b)
            right = self.difference_expert((a - b).abs())
        elif self.policy['interaction_mode'] == 'matched_eeg_meg_concat':
            left = self.product_expert(a)
            right = self.difference_expert(b)
        else:
            raise AssertionError(self.policy['interaction_mode'])
        return self.pair_projection(torch.cat((left * expert_weights[..., :1], right * expert_weights[..., 1:]), dim=-1))

    def _path_residuals(self, eeg: torch.Tensor, meg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg_e2m = self.coordinates.path_features(eeg, PATH_EEG_TO_MEG, MODALITY_EEG)
        meg_e2m = self.coordinates.path_features(meg, PATH_EEG_TO_MEG, MODALITY_MEG)
        meg_m2e = self.coordinates.path_features(meg, PATH_MEG_TO_EEG, MODALITY_MEG)
        eeg_m2e = self.coordinates.path_features(eeg, PATH_MEG_TO_EEG, MODALITY_EEG)
        e2m_input = torch.cat((eeg_e2m, meg_e2m, eeg_e2m * meg_e2m, (eeg_e2m - meg_e2m).abs()), dim=-1)
        m2e_input = torch.cat((meg_m2e, eeg_m2e, meg_m2e * eeg_m2e, (meg_m2e - eeg_m2e).abs()), dim=-1)
        e2m = self.path_eeg_to_meg(e2m_input)
        m2e = self.path_meg_to_eeg(m2e_input)
        if not bool(self.policy['path_enabled']):
            e2m = torch.zeros_like(e2m)
            m2e = torch.zeros_like(m2e)
        policy = torch.ones((len(eeg), WINDOWS, 2), device=eeg.device, dtype=eeg.dtype)
        if not bool(self.policy['path_enabled']):
            policy.zero_()
        return (e2m, m2e, policy)

    def _temporal_weights(self, role_support: torch.Tensor, raw_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.policy['temporal_mode'] in {'exact_uniform', 'exact_uniform_without_controller'}:
            zeros = torch.zeros_like(role_support)
            return (torch.ones_like(role_support), zeros, zeros, zeros)
        public_logits = TEMPORAL_PUBLIC_SCALE * role_support.clamp_min(EPS).log()
        learned = TEMPORAL_RESIDUAL_SCALE * torch.tanh(self.temporal_controller_residual(raw_state))
        logits = public_logits + learned
        weights = torch.softmax(logits.transpose(1, 2), dim=-1).transpose(1, 2) * float(WINDOWS)
        return (weights, logits, public_logits, learned)

    def _branch_weights(self, branch_marker: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.policy['branch_mode'] == 'exact_quarters':
            batch = len(branch_marker)
            fixed = torch.full((batch, BRANCHES), 1.0 / float(BRANCHES), device=branch_marker.device, dtype=branch_marker.dtype)
            zeros = torch.zeros_like(fixed)
            return (fixed, zeros, zeros)
        public = BRANCH_PUBLIC_SCALE * self._within_feature_z(branch_marker.mean(dim=1))
        learned = BRANCH_RESIDUAL_SCALE * torch.tanh(self.branch_marker_residual(branch_marker.mean(dim=1)))
        return (torch.softmax(public + learned, dim=-1), public, learned)

    def _anchor_feature(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        eeg_feature = self.coordinates.anchor_features(eeg, MODALITY_EEG).mean(dim=1)
        meg_feature = self.coordinates.anchor_features(meg, MODALITY_MEG).mean(dim=1)
        return torch.cat((eeg_feature, meg_feature), dim=-1)

    def anchor_log_probability(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        feature = self._anchor_feature(eeg, meg)
        return torch.log_softmax(self.anchor(feature), dim=-1)

    def freeze_anchor(self) -> None:
        self._anchor_frozen = True
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.anchor.eval()

    def _mix_weight(self, eeg: torch.Tensor, meg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(self.anchor_mixing_calibration_fitted.item()):
            raise RuntimeError('V25 Anchor-route calibration is required before inference')
        (cue, marker) = self._route_cue(eeg, meg)
        public = self._public_structured_weight(cue)
        if self.policy['anchor_mixing_mode'] == 'fixed_outer_train_lambda':
            fixed = self.anchor_fixed_lambda.to(dtype=eeg.dtype, device=eeg.device).expand(len(eeg), 1)
            return (fixed, marker, public, torch.zeros_like(fixed))
        residual = MIX_RESIDUAL_SCALE * torch.tanh(self.anchor_mix_residual(marker.mean(dim=1)))
        adaptive = (public + residual).clamp(MIX_MIN, MIX_MIN + MIX_RANGE)
        return (adaptive, marker, public, residual)

    def forward_with_details(self, eeg: torch.Tensor, meg: torch.Tensor) -> dict[str, torch.Tensor]:
        _CoordinatePowerBank._validate(eeg)
        _CoordinatePowerBank._validate(meg)
        raw_state = self._raw_controller_state(eeg, meg)
        membership = self._controller_membership(raw_state)
        reported_state = raw_state if bool(self.policy['controller_enabled']) else torch.zeros_like(raw_state)
        role_support = self._role_support(membership)
        if not bool(self.policy['controller_enabled']):
            role_support = torch.ones_like(role_support)
        eeg_direct = self.eeg_direct_encoder(self.coordinates.direct_features(eeg, MODALITY_EEG))
        meg_direct = self.meg_direct_encoder(self.coordinates.direct_features(meg, MODALITY_MEG))
        expert_weights = self._interaction_weights(membership, raw_state)
        pair_token = self._pair_token(self.coordinates.pair_features(eeg, MODALITY_EEG), self.coordinates.pair_features(meg, MODALITY_MEG), expert_weights)
        (e2m_residual, m2e_residual, exchange_policy) = self._path_residuals(eeg, meg)
        fusion_input = torch.cat((e2m_residual, m2e_residual), dim=-1)
        fusion_token = self.fusion_projection(fusion_input)
        (local_weights, temporal_logits, temporal_public, temporal_learned) = self._temporal_weights(role_support, raw_state)
        summaries = torch.stack((self._weighted_mean(eeg_direct, local_weights[..., BRANCH_EEG]), self._weighted_mean(meg_direct, local_weights[..., BRANCH_MEG]), self._weighted_mean(pair_token, local_weights[..., BRANCH_PAIR]), self._weighted_mean(fusion_token, local_weights[..., BRANCH_FUSION])), dim=1)
        branch_logits = torch.stack((self.eeg_head(summaries[:, BRANCH_EEG]), self.meg_head(summaries[:, BRANCH_MEG]), self.pair_head(summaries[:, BRANCH_PAIR]), self.fusion_head(summaries[:, BRANCH_FUSION])), dim=1)
        branch_marker = self.coordinates.branch_marker(eeg, meg)
        (branch_weights, branch_public, branch_learned) = self._branch_weights(branch_marker)
        structured_probability = torch.softmax((branch_weights.unsqueeze(-1) * branch_logits).sum(dim=1), dim=-1)
        anchor_feature = self._anchor_feature(eeg, meg)
        anchor_probability = torch.softmax(self.anchor(anchor_feature), dim=-1)
        (mix_weight, anchor_route, anchor_public, mix_residual) = self._mix_weight(eeg, meg)
        probability = mix_weight * structured_probability + (1.0 - mix_weight) * anchor_probability
        return {'log_probability': probability.clamp_min(EPS).log(), 'local_weights': local_weights, 'branch_weights': branch_weights, 'mix_weight': mix_weight, 'structured_probability': structured_probability, 'anchor_probability': anchor_probability, 'direct_eeg_token': eeg_direct, 'direct_meg_token': meg_direct, 'pair_token': pair_token, 'path_eeg_to_meg_residual': e2m_residual, 'path_meg_to_eeg_residual': m2e_residual, 'fusion_input': fusion_input, 'fusion_token': fusion_token, 'exchange_policy': exchange_policy, 'controller_state': reported_state, 'controller_membership': membership, 'controller_role_support': role_support, 'interaction_weights': expert_weights, 'temporal_logits': temporal_logits, 'temporal_public_logits': temporal_public, 'temporal_learned_residual': temporal_learned, 'temporal_weights': local_weights, 'branch_marker': branch_marker, 'branch_public_logits': branch_public, 'branch_learned_residual': branch_learned, 'branch_logits': branch_logits, 'anchor_task_feature': anchor_feature, 'anchor_route_feature': anchor_route, 'anchor_public_cue': anchor_public, 'anchor_pretraining_weight': self.anchor_pretraining_weight(eeg, meg), 'anchor_fixed_lambda': self.anchor_fixed_lambda.to(dtype=eeg.dtype, device=eeg.device).expand(len(eeg), 1), 'mix_residual': mix_residual}

    def forward(self, eeg: torch.Tensor, meg: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(eeg, meg)['log_probability']

def make_model(variant: str=FULL_VARIANT, dropout: float=0.2) -> nn.Module:
    if variant == EEG_ONLY:
        return _SingleCoordinateModelV25(MODALITY_EEG)
    if variant == MEG_ONLY:
        return _SingleCoordinateModelV25(MODALITY_MEG)
    return HierarchicalPairAnchorModelV25(variant=variant, dropout=dropout)

def _add_axis(values: torch.Tensor, axis: torch.Tensor, *, amplitude: float=0.75) -> torch.Tensor:
    time = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype)
    carrier = torch.sin(2.0 * math.pi * 13.0 * time / 250.0)
    return values + float(amplitude) * axis.to(values).view(1, CHANNELS, 1) * carrier.view(1, 1, -1)

def _assert_same(left: torch.Tensor, right: torch.Tensor, label: str, *, atol: float=2e-05) -> None:
    if not torch.allclose(left, right, atol=atol, rtol=atol):
        raise AssertionError(label)

def _assert_changed(left: torch.Tensor, right: torch.Tensor, label: str, *, minimum: float=2e-06) -> None:
    if float((left - right).abs().max()) <= minimum:
        raise AssertionError(label)

def _assert_fusion_boundary(detail: dict[str, torch.Tensor]) -> None:
    fusion = detail['fusion_input']
    assert torch.equal(fusion[..., :TOKEN_DIM], detail['path_eeg_to_meg_residual'])
    assert torch.equal(fusion[..., TOKEN_DIM:], detail['path_meg_to_eeg_residual'])
    assert fusion.shape[-1] == 2 * TOKEN_DIM

def _band_selectivity_check(bank: _CoordinatePowerBank) -> None:
    axis = bank.direct_task[MODALITY_EEG, 0, 0]
    duplicated_maps = axis.view(1, 1, CHANNELS).expand(BANDS, 1, CHANNELS)
    time = torch.arange(WINDOWS * SAMPLES_PER_WINDOW, dtype=axis.dtype, device=axis.device) / SAMPLE_RATE_HZ

    def tone(frequency: float) -> torch.Tensor:
        return 1.5 * axis.view(1, CHANNELS, 1) * torch.sin(2.0 * math.pi * float(frequency) * time).view(1, 1, -1)
    with torch.no_grad():
        low = bank._window_log_power(tone(10.0), duplicated_maps)
        high = bank._window_log_power(tone(21.0), duplicated_maps)
    if not float(low[..., 0].mean() - low[..., 1].mean()) > 8.0:
        raise AssertionError('V25 10 Hz band-selectivity check failed')
    if not float(high[..., 1].mean() - high[..., 0].mean()) > 8.0:
        raise AssertionError('V25 21 Hz band-selectivity check failed')

def self_test() -> dict[str, object]:
    torch.set_num_threads(2)
    torch.manual_seed(1025)
    eeg = torch.randn(5, CHANNELS, WINDOWS * SAMPLES_PER_WINDOW)
    meg = torch.randn(5, CHANNELS, WINDOWS * SAMPLES_PER_WINDOW)
    models: dict[str, nn.Module] = {}
    for variant in VARIANTS:
        torch.manual_seed(1025)
        models[variant] = make_model(variant, dropout=0.0).eval()
    structured_counts = {variant: parameter_count(models[variant]) for variant in STRUCTURED_VARIANTS}
    if len(set(structured_counts.values())) != 1:
        raise AssertionError(structured_counts)
    reference = models[FULL_VARIANT].state_dict()
    for variant in STRUCTURED_VARIANTS:
        state = models[variant].state_dict()
        if tuple(state) != tuple(reference) or not all((torch.equal(state[key], reference[key]) for key in state)):
            raise AssertionError(f'V25 initial-state contract failed: {variant}')
    full_unfitted = models[FULL_VARIANT]
    assert isinstance(full_unfitted, HierarchicalPairAnchorModelV25)
    try:
        full_unfitted.forward_with_details(eeg, meg)
        raise AssertionError('Uncalibrated V25 inference unexpectedly succeeded')
    except RuntimeError as error:
        if 'calibration' not in str(error).lower():
            raise
    for variant in STRUCTURED_VARIANTS:
        model = models[variant]
        assert isinstance(model, HierarchicalPairAnchorModelV25)
        model.fit_controller_calibration(eeg, meg)
        model.fit_anchor_mixing_calibration(eeg, meg)
    with torch.no_grad():
        for (variant, model) in models.items():
            detail = model.forward_with_details(eeg, meg)
            probability = detail['log_probability'].exp()
            assert torch.isfinite(probability).all()
            _assert_same(probability.sum(-1), torch.ones(len(eeg)), f'{variant}.probability')
            if variant in SINGLE_VARIANTS:
                continue
            assert set(DETAIL_KEYS).issubset(detail)
            assert detail['local_weights'].shape == (len(eeg), WINDOWS, BRANCHES)
            _assert_same(detail['local_weights'].sum(dim=1), torch.full((len(eeg), BRANCHES), float(WINDOWS)), f'{variant}.temporal_sum')
            _assert_same(detail['branch_weights'].sum(dim=-1), torch.ones(len(eeg)), f'{variant}.branch_sum')
            _assert_same(detail['interaction_weights'].sum(dim=-1), torch.ones(len(eeg), WINDOWS), f'{variant}.expert_sum')
            _assert_fusion_boundary(detail)
            if variant == UNIFORM_TEMPORAL:
                assert torch.equal(detail['local_weights'], torch.ones_like(detail['local_weights']))
            if variant == FIXED_BRANCH_GATE:
                assert torch.equal(detail['branch_weights'], torch.full_like(detail['branch_weights'], 0.25))
            if variant == FIXED_ANCHOR_MIXING:
                assert torch.equal(detail['mix_weight'], detail['anchor_fixed_lambda'])
    full = models[FULL_VARIANT]
    no_path_controller = models[NO_BIDIRECTIONAL_PATH_CONTROLLER]
    assert isinstance(full, HierarchicalPairAnchorModelV25)
    assert isinstance(no_path_controller, HierarchicalPairAnchorModelV25)
    _band_selectivity_check(full.coordinates)
    with torch.no_grad():
        baseline = full.forward_with_details(eeg, meg)
        no_bundle = no_path_controller.forward_with_details(eeg, meg)
    _assert_fusion_boundary(baseline)
    assert torch.equal(no_bundle['path_eeg_to_meg_residual'], torch.zeros_like(no_bundle['path_eeg_to_meg_residual']))
    assert torch.equal(no_bundle['path_meg_to_eeg_residual'], torch.zeros_like(no_bundle['path_meg_to_eeg_residual']))
    assert torch.equal(no_bundle['exchange_policy'], torch.zeros_like(no_bundle['exchange_policy']))
    assert torch.equal(no_bundle['controller_state'], torch.zeros_like(no_bundle['controller_state']))
    assert torch.equal(no_bundle['local_weights'], torch.ones_like(no_bundle['local_weights']))
    assert torch.equal(no_bundle['interaction_weights'], torch.full_like(no_bundle['interaction_weights'], 0.5))
    arrays = registry_arrays(copy=False)
    controller_meg = _add_axis(meg, torch.as_tensor(arrays['controller'][MODALITY_MEG, CONTROLLER_PAIR_EXPERT], dtype=meg.dtype))
    branch_eeg = _add_axis(eeg, torch.as_tensor(arrays['branch'][MODALITY_EEG, BRANCH_PAIR], dtype=eeg.dtype))
    route_meg = _add_axis(meg, torch.as_tensor(arrays['anchor_route'][MODALITY_MEG, ANCHOR_ROUTE_STRUCTURED], dtype=meg.dtype))
    anchor_eeg = _add_axis(eeg, torch.as_tensor(arrays['anchor_task'][MODALITY_EEG, 1], dtype=eeg.dtype))
    pair_eeg = _add_axis(eeg, torch.as_tensor(arrays['pair_task'][MODALITY_EEG, EXPERT_PRODUCT, 0], dtype=eeg.dtype))
    path_eeg = _add_axis(eeg, torch.as_tensor(arrays['path_task'][PATH_EEG_TO_MEG, path_endpoint_for_modality(PATH_EEG_TO_MEG, MODALITY_EEG), 0], dtype=eeg.dtype))
    with torch.no_grad():
        controller_changed = full.forward_with_details(eeg, controller_meg)
        branch_changed = full.forward_with_details(branch_eeg, meg)
        route_changed = full.forward_with_details(eeg, route_meg)
        anchor_changed = full.forward_with_details(anchor_eeg, meg)
        pair_changed = full.forward_with_details(pair_eeg, meg)
        path_changed = full.forward_with_details(path_eeg, meg)
    _assert_changed(baseline['interaction_weights'], controller_changed['interaction_weights'], 'controller must reach expert selector')
    _assert_changed(baseline['local_weights'], controller_changed['local_weights'], 'controller must reach temporal router')
    for key in ('direct_eeg_token', 'direct_meg_token', 'fusion_token', 'path_eeg_to_meg_residual', 'path_meg_to_eeg_residual', 'branch_weights', 'anchor_probability', 'mix_weight'):
        _assert_same(baseline[key], controller_changed[key], f'controller leaked into {key}')
    _assert_changed(baseline['branch_weights'], branch_changed['branch_weights'], 'branch marker must reach branch gate')
    for key in ('direct_eeg_token', 'direct_meg_token', 'pair_token', 'path_eeg_to_meg_residual', 'path_meg_to_eeg_residual', 'local_weights', 'anchor_probability', 'mix_weight'):
        _assert_same(baseline[key], branch_changed[key], f'branch marker leaked into {key}')
    _assert_changed(baseline['mix_weight'], route_changed['mix_weight'], 'Anchor route must reach mixing')
    _assert_changed(baseline['anchor_pretraining_weight'], route_changed['anchor_pretraining_weight'], 'Anchor route must reach pretraining weight')
    for key in ('direct_eeg_token', 'direct_meg_token', 'pair_token', 'path_eeg_to_meg_residual', 'path_meg_to_eeg_residual', 'branch_weights', 'structured_probability', 'anchor_probability'):
        _assert_same(baseline[key], route_changed[key], f'Anchor route leaked into {key}')
    _assert_changed(baseline['anchor_probability'], anchor_changed['anchor_probability'], 'Anchor task must reach Anchor classifier')
    for key in ('direct_eeg_token', 'direct_meg_token', 'pair_token', 'path_eeg_to_meg_residual', 'path_meg_to_eeg_residual', 'branch_weights', 'structured_probability', 'mix_weight'):
        _assert_same(baseline[key], anchor_changed[key], f'Anchor task leaked into {key}')
    _assert_changed(baseline['pair_token'], pair_changed['pair_token'], 'pair axis must reach paired experts')
    for key in ('direct_eeg_token', 'direct_meg_token', 'fusion_token', 'path_eeg_to_meg_residual', 'path_meg_to_eeg_residual', 'controller_state', 'branch_weights', 'anchor_probability', 'mix_weight'):
        _assert_same(baseline[key], pair_changed[key], f'pair task leaked into {key}')
    _assert_changed(baseline['path_eeg_to_meg_residual'], path_changed['path_eeg_to_meg_residual'], 'E->M path axis must reach its residual')
    for key in ('direct_eeg_token', 'direct_meg_token', 'pair_token', 'path_meg_to_eeg_residual', 'controller_state', 'branch_weights', 'anchor_probability', 'mix_weight'):
        _assert_same(baseline[key], path_changed[key], f'path task leaked into {key}')
    with torch.no_grad():
        no_controller_before = no_path_controller.forward_with_details(eeg, meg)
        no_controller_after = no_path_controller.forward_with_details(eeg, controller_meg)
        no_path_after = no_path_controller.forward_with_details(path_eeg, meg)
    _assert_same(no_controller_before['interaction_weights'], no_controller_after['interaction_weights'], 'disabled controller changed expert selector')
    _assert_same(no_controller_before['local_weights'], no_controller_after['local_weights'], 'disabled controller changed temporal routes')
    _assert_same(no_path_after['path_eeg_to_meg_residual'], torch.zeros_like(no_path_after['path_eeg_to_meg_residual']), 'disabled path survived')
    eeg_mono = models[EEG_ONLY]
    meg_mono = models[MEG_ONLY]
    with torch.no_grad():
        assert torch.equal(eeg_mono(eeg, meg), eeg_mono(eeg, meg + 5.0))
        assert torch.equal(meg_mono(eeg, meg), meg_mono(eeg + 5.0, meg))
    full.freeze_anchor()
    full.train()
    if full.anchor.training or any((parameter.requires_grad for parameter in full.anchor.parameters())):
        raise AssertionError('Frozen V25 Anchor did not remain frozen')
    return {'status': 'PASS', 'model_family': MODEL_FAMILY, 'coordinate_registry_sha256': REGISTRY_SHA256, 'parameter_count': structured_counts, 'structured_trainable_after_anchor_freeze': trainable_parameter_count(full), 'primary_ablations': PRIMARY_ABLATIONS, 'graph_contract': GRAPH_CONTRACT}
if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='V25 coordinate-isolation model self-test')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Only --self-test is supported')
    print(json.dumps(self_test(), indent=2, sort_keys=True))
