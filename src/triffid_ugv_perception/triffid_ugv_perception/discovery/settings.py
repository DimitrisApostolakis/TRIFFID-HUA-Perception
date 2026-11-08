"""Explicit resource and confirmation limits for a mission."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class DiscoverySettings:
    updates_enabled: bool = False
    update_interval_s: float = 2.
    update_distance_m: float = 1.
    update_altitude_m: float = 1.
    update_confidence_delta: float = .1
    class_correction: bool = False
    namespace: str = 'PUI/front'
    embedding_space: str = 'none'
    confirm_frames: int = 2
    confirmation_gap_s: float = 2.5
    min_confidence: float = .35
    alias_ttl_s: float = 60.
    max_candidates: int = 512
    max_objects: int = 10000
    max_pending: int = 1000
    max_pending_bytes: int = 8 * 1024 * 1024
    max_payload_bytes: int = 16 * 1024
    max_database_bytes: int = 64 * 1024 * 1024
    max_appearance_dims: int = 512
    reid_similarity: float = .90
    reid_uncertain_similarity: float = .75
    reid_margin: float = .05
    reid_iou: float = .15
    reid_radius_m: float = 3.
    flush_interval_s: float = 5.
    lease_s: float = 30.
    retry_initial_s: float = 1.
    retry_max_s: float = 60.
    log_bytes: int = 1024 * 1024
    log_backups: int = 2

    def __post_init__(self):
        if any(not isinstance(v,str) or not v.strip() or len(v)>128 for v in (self.namespace,self.embedding_space)):
            raise ValueError('A stable mission/camera namespace and embedding space are required')
        for field in fields(self):
            value = getattr(self,field.name)
            if isinstance(value,float) and not math.isfinite(value):
                raise ValueError('Settings must be finite')
        integer_fields = ('confirm_frames', 'max_candidates', 'max_objects', 'max_pending',
                          'max_pending_bytes', 'max_payload_bytes', 'max_database_bytes',
                          'max_appearance_dims', 'log_bytes', 'log_backups')
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in integer_fields):
            raise ValueError('Storage/count limits must be positive integers')
        if self.max_database_bytes < 256 * 1024:
            raise ValueError('Database/journal budget must be at least 256 KiB')
        if not 0 <= self.reid_uncertain_similarity <= self.reid_similarity <= 1:
            raise ValueError('Invalid re-identification thresholds')
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.reid_iou <= 1:
            raise ValueError('Invalid confidence/overlap threshold')
        if any(getattr(self, k) <= 0 for k in ('update_interval_s', 'update_distance_m', 'update_altitude_m', 'update_confidence_delta', 'confirmation_gap_s', 'alias_ttl_s', 'flush_interval_s',
                'lease_s', 'retry_initial_s', 'retry_max_s', 'reid_radius_m')):
            raise ValueError('Timing/distance limits must be positive')
        if self.retry_max_s < self.retry_initial_s or not 0 <= self.reid_margin <= 1:
            raise ValueError('Invalid retry/re-identification limits')
