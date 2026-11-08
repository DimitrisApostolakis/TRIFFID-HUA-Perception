"""Select distinct angular views from telemetry before decoding any video frames."""
import math
from dataclasses import asdict, dataclass

from .srt_metadata import SrtFrameMeta


def angular_distance(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class SelectedView:
    view_id: str
    frame_index: int
    timestamp_s: float
    target_angle_deg: float
    actual_angle_deg: float
    angular_error_deg: float
    target_pitch_deg: float | None
    telemetry: SrtFrameMeta

    def to_dict(self) -> dict:
        return asdict(self)


def select_views(records: list[SrtFrameMeta], fps: float, frame_count: int, *,
                 views: int = 8, mode: str = 'orbit', start_s: float = 0.0,
                 end_s: float | None = None, angle_offset: float = 0.0,
                 tolerance: float = 12.0, centre: tuple[float, float] | None = None,
                 pitches: tuple[float, ...] | None = None,
                 pitch_tolerance: float = 10.0) -> tuple[list[SelectedView], dict]:
    """Angles clockwise from north. Orbit angle is drone position about the centre,
    not camera yaw. Use a time window containing one circle. SRT timestamps, not
    subtitle sequence numbers, map lower-rate telemetry to video frames.
    """
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError('Video must provide positive FPS and frame count')
    if not 1 <= views <= 360:
        raise ValueError('views must be between 1 and 360')
    if mode not in ('orbit', 'heading'):
        raise ValueError('selection mode must be orbit or heading')
    values = [start_s, angle_offset, tolerance, pitch_tolerance]
    values += list(centre or ()) + list(pitches or ())
    if end_s is not None:
        values.append(end_s)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Selection parameters must be finite')
    if start_s < 0 or (end_s is not None and end_s <= start_s):
        raise ValueError('Use 0 <= start_s < end_s')
    if not 0 <= tolerance < 180 or pitch_tolerance < 0:
        raise ValueError('Invalid angular tolerance')
    if pitches is not None and (not pitches or len(set(pitches)) != len(pitches)):
        raise ValueError('Pitch targets must be nonempty and distinct')
    records = [r for r in records if math.isfinite(r.timestamp_s)
               and start_s <= r.timestamp_s and (end_s is None or r.timestamp_s < end_s)
               and 0 <= round(r.timestamp_s * fps) < frame_count]
    if mode == 'orbit':
        records = [r for r in records if r.has_position and
                   all(math.isfinite(v) for v in (r.lat, r.lon))]
        if not records:
            raise ValueError('No usable GPS telemetry in the selected time window')
        if centre is None:
            # Bounding-box centre is independent of pauses/different orbit speeds.
            centre = ((min(r.lat for r in records) + max(r.lat for r in records)) / 2,
                      (min(r.lon for r in records) + max(r.lon for r in records)) / 2)
            centre_source = 'gps_bounds_estimate'
        else:
            centre_source = 'explicit'
        lat, lon = centre
        cosine = math.cos(math.radians(lat))
        def angle(r):
            return math.degrees(math.atan2((r.lon - lon) * cosine, r.lat - lat)) % 360
        records = [r for r in records if math.hypot((r.lon - lon) * cosine, r.lat - lat) > 1e-9]
    else:
        records = [r for r in records if r.gimbal_yaw is not None and math.isfinite(r.gimbal_yaw)]
        def angle(r):
            return r.gimbal_yaw % 360
        centre_source = None

    # Assignment is global over all angle/pitch targets. At most 16 small target
    # lists; resolve collisions using augmenting paths instead of reusing a frame.
    targets = [(p, (angle_offset + i * 360 / views) % 360)
               for p in (pitches if pitches is not None else (None,)) for i in range(views)]
    candidates = []
    for pitch, target in targets:
        by_frame = {}
        for r in records:
            if pitch is not None and (r.gimbal_pitch is None or
                    not math.isfinite(r.gimbal_pitch) or abs(r.gimbal_pitch - pitch) > pitch_tolerance):
                continue
            error = angular_distance(angle(r), target)
            if error <= tolerance:
                frame = round(r.timestamp_s * fps)
                rank = (error, abs(r.gimbal_pitch - pitch) if pitch is not None else 0,
                        abs(frame / fps - r.timestamp_s), r.timestamp_s)
                if frame not in by_frame or rank < by_frame[frame][0]:
                    by_frame[frame] = (rank, r)
        candidates.append(sorted(by_frame.items(), key=lambda pair: pair[1][0]))
    owners, assigned = {}, {}
    def assign(target_index, seen):
        for frame, (_, record) in candidates[target_index]:
            if frame in seen:
                continue
            seen.add(frame)
            if frame not in owners or assign(owners[frame], seen):
                owners[frame] = target_index
                assigned[target_index] = (frame, record)
                return True
        return False
    for i in sorted(range(len(targets)), key=lambda i: len(candidates[i])):
        if not assign(i, set()):
            pitch, target = targets[i]
            raise ValueError(f'Cannot select {len(targets)} distinct views: missing angle '
                             f'{target:g} degrees, pitch {pitch}. Check circle/time window/tolerances.')
    selected = []
    for i, (pitch, target) in enumerate(targets):
        frame, record = assigned[i]
        actual = angle(record)
        selected.append(SelectedView(f'view_{i:02d}', frame, frame / fps, target, actual,
                                     angular_distance(actual, target), pitch, record))
    return selected, {'mode': mode, 'views_per_pitch': views,
                      'angle_offset_deg': angle_offset, 'angle_tolerance_deg': tolerance,
                      'pitch_targets_deg': pitches, 'pitch_tolerance_deg': pitch_tolerance,
                      'start_s': start_s, 'end_s': end_s,
                      'orbit_centre_lat_lon': centre, 'centre_source': centre_source}
