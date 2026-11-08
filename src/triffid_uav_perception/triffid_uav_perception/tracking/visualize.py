"""Small tracking artifacts: three videos and JPEGs, without HTML or image copies."""
from pathlib import Path

import cv2

from ..visualize import class_color


class TrackingRenderer:
    def __init__(self, output, fps=1):
        if fps <= 0:
            raise ValueError('Preview FPS must be positive')
        self.output, self.fps = Path(output), fps
        self.writers, self.size = {}, None
        self.index = 0

    def __call__(self, image, source, tracks):
        self.output.mkdir(parents=True, exist_ok=True)
        size = image.shape[1], image.shape[0]
        if self.size is not None and self.size != size:
            raise ValueError('Tracking preview video requires constant frame dimensions')
        self.size = size
        groups = dict(detections=source['observations'], tracking=tracks,
                      exports=[o for o in tracks if o['tracking']['is_new']])
        for name, observations in groups.items():
            canvas = image.copy()
            for obs in observations:
                rgb = class_color(obs.get('tracking', {}).get('object_id', obs['class_name']))
                color = tuple(reversed(rgb))
                x1, y1, x2, y2 = (int(v) for v in obs['bbox_xyxy'])
                cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), color, 2)
                score = obs.get('score')
                label = obs['class_name'] + (f' {score:.0%}' if score is not None else '')
                if 'tracking' in obs:
                    track = obs['tracking']
                    marker = '*' if track['state'] == 'tentative' else ''
                    label = f'T{track["track_id"]}{marker} ' + label
                cv2.putText(canvas, label, (x1, max(38, y1-5)), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, color, 1, cv2.LINE_AA)
            caption = (f'{name}: {len(observations)} | source {source["source_id"]} '
                       f't={source["timestamp_s"]:.3f}s | '
                       f'{source.get("sampling_description", "selected-frame preview")} | * tentative')
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (15, 15, 15), -1)
            cv2.putText(canvas, caption, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            folder = self.output/'overlays'/name
            folder.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(folder/f'{self.index:04d}.jpg'), canvas):
                raise OSError('Could not write tracking overlay')
            if name not in self.writers:
                writer = cv2.VideoWriter(str(self.output/f'{name}.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size)
                if not writer.isOpened():
                    writer.release()
                    raise OSError('Could not create tracking preview video')
                self.writers[name] = writer
            self.writers[name].write(canvas)
        self.index += 1

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
