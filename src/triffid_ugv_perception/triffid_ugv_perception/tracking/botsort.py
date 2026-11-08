"""BoT-SORT wrapper for the 1 Hz baseline, using existing detector descriptors."""
from types import SimpleNamespace
import numpy as np


class BotSortTracker:
    def __init__(self, fps=1, lost_seconds=10):
        if fps<=0 or lost_seconds<0:
            raise ValueError('Invalid tracking frame rate/lost interval')
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.engine.results import Boxes
        class IndexedBoxes(Boxes):
            """Retain input row identity through high/low confidence slicing."""
            def __init__(self,data,orig_shape,source_indices=None):
                super().__init__(data,orig_shape)
                self.source_indices=np.arange(len(self)) if source_indices is None else np.atleast_1d(source_indices)

            def __getitem__(self,index):
                return type(self)(self.data[index],self.orig_shape,self.source_indices[index])
        # Ultralytics scales track_buffer by frame_rate/30. At 1 Hz, this
        # preserves lost_seconds seconds rather than the default one-frame buffer.
        settings=SimpleNamespace(track_high_thresh=.35,track_low_thresh=.1,new_track_thresh=.4,
            track_buffer=round(30*lost_seconds),match_thresh=.8,fuse_score=True,
            gmc_method='sparseOptFlow',proximity_thresh=.5,appearance_thresh=.25,
            with_reid=True,model='auto')
        self.next_id=1
        def allocate_id():
            result=self.next_id
            self.next_id+=1
            return result

        class ClassAwareBoTSORT(BOTSORT):
            @staticmethod
            def reset_id():
                # Allocation belongs to this wrapper, not Ultralytics globals.
                pass

            def init_track(self,results,img=None):
                tracks=super().init_track(results,img)
                # Ultralytics numbers each filtered confidence tier from zero.
                # Expose the actual input row, including for low-score recovery.
                for track,index in zip(tracks,results.source_indices):
                    track.idx=int(index)
                    track.next_id=allocate_id
                return tracks

            def get_dists(self, tracks, detections):
                distances=super().get_dists(tracks,detections)
                for i,track in enumerate(tracks):
                    for j,detection in enumerate(detections):
                        if track.cls != detection.cls:
                            distances[i,j]=1e5
                return distances

            @staticmethod
            def remove_duplicate_stracks(active,lost):
                # The upstream duplicate-removal stage is class-agnostic too.
                kept_active,kept_lost=[],[]
                for cls in {t.cls for t in active+lost}:
                    a,b=BOTSORT.remove_duplicate_stracks([t for t in active if t.cls==cls],
                                                       [t for t in lost if t.cls==cls])
                    kept_active.extend(a);kept_lost.extend(b)
                return kept_active,kept_lost
        self.tracker=ClassAwareBoTSORT(settings,frame_rate=fps)
        self.boxes_type=IndexedBoxes
        self.settings=settings
        self.fps=fps
        self.gmc=self.tracker.gmc
        self.current_warp=np.eye(2,3)
        owner=self
        class SharedCameraMotion:
            def apply(self,*args,**kwargs):
                return owner.current_warp
        self.shared_gmc=SharedCameraMotion()
        self.make_tracker=lambda:ClassAwareBoTSORT(settings,frame_rate=fps)
        self.pools={}
        self.calls=0
        self.frame_clock=0
        self.last_timestamp=None

    def metadata(self):
        return dict(name='botsort',settings=vars(self.settings),fps=self.fps,
                    adapter_revision='20261106_class_pools_and_source_v3',
                    association='class-compatible pools in all stages; shared camera motion',
                    clock='timestamp gaps advance missing ticks without detections',
                    appearance='supplied detector descriptors; no separate learned ReID model')

    def update(self,detections,bgr,timestamp_ns=None):
        import torch
        timestamp=int(timestamp_ns) if timestamp_ns is not None else round(self.calls*1e9/self.fps)
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:
            raise ValueError('Tracking timestamps must increase')
        ticks=1 if self.last_timestamp is None else max(1,round((timestamp-self.last_timestamp)*self.fps/1e9))
        skipped=min(ticks-1,self.tracker.max_time_lost+2)
        empty=self.boxes_type(np.empty((0,6),dtype=np.float32),bgr.shape[:2])
        for _ in range(skipped):
            for engine in self.pools.values():
                engine.update(empty,img=None,feats=None)
        self.frame_clock+=skipped
        self.last_timestamp=timestamp
        boxes=np.asarray([[*d['bbox'],d['confidence'],d['class_id']] for d in detections],dtype=np.float32).reshape(-1,6)
        features=torch.as_tensor(np.asarray([d['appearance'] for d in detections],dtype=np.float32)) if detections else None
        # Ultralytics' second (weak-score) stage bypasses get_dists and is
        # class-agnostic. Separate pools enforce compatibility in every stage.
        # The costly camera-motion estimate is computed only once per image.
        try:
            self.current_warp=self.gmc.apply(bgr,boxes[boxes[:,4]>=self.settings.track_high_thresh,:4])
        except Exception:
            self.current_warp=np.eye(2,3)
        for cid in dict.fromkeys(int(d['class_id']) for d in detections):
            if cid not in self.pools:
                engine=self.tracker if not self.pools else self.make_tracker()
                engine.gmc=self.shared_gmc
                engine.frame_id=self.frame_clock
                self.pools[cid]=engine
        tracks=[]
        for cid,engine in self.pools.items():
            indices=np.flatnonzero(boxes[:,5]==cid)
            inputs=self.boxes_type(boxes[indices],bgr.shape[:2],indices)
            result=engine.update(inputs,img=bgr,feats=features[indices] if features is not None else None)
            for row in result:
                x1,y1,x2,y2,tid,score,predicted_class,source=row
                original=detections[int(source)]
                assert int(predicted_class)==cid==original['class_id']
                tracks.append(dict(original,bbox=list(map(float,(x1,y1,x2,y2))),track_id=int(tid),
                    confidence=float(score),class_id=cid,class_name=original['class_name'],
                    appearance=original['appearance']))
        self.calls+=1
        self.frame_clock+=1
        return tracks
