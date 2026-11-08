"""Detector-independent UDC adapter calibrated for a one-Hz input stream."""
from dataclasses import asdict, replace
import numpy as np
from .vendor.udc_core import ByteTracker, Settings


class UdcTracker:
    def __init__(self, fps=1, lost_seconds=10, proof_threshold=.35):
        if fps != 1 or lost_seconds < 0:
            raise ValueError('UDC adapter is calibrated for 1 Hz and nonnegative lost_seconds')
        if not .35<=proof_threshold<=1:
            raise ValueError('UDC proof threshold must be between 0.35 and 1')
        self.settings=replace(Settings(),max_age=int(lost_seconds),proof_thr=float(proof_threshold))
        self.pools={};self.identities={};self.next_id=1;self.calls=0
        self.last_timestamp=None;self.revivals=[]

    def metadata(self):
        return dict(name='udc',revision='11188042c8b487374a75b609424998df3f041522',
                    settings=asdict(self.settings),clock='1-second ticks; missing ticks advance without detections',
                    appearance='HSV histogram; correlation >= 0.9 for gallery recovery',
                    adapter_revision='20261106_assignment_and_source_v2',
                    assignment='IoU gate inside global assignment; exact matched detection metadata')

    def update(self,detections,bgr,timestamp_ns=None):
        timestamp=int(timestamp_ns) if timestamp_ns is not None else self.calls*10**9
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Tracking timestamps must increase')
        # Advance missing one-second observations, bounded after all memory expires.
        ticks=1 if self.last_timestamp is None else max(1,round((timestamp-self.last_timestamp)/1e9))
        skipped=min(ticks-1,self.settings.max_age+self.settings.gallery_ttl+2)
        for _ in range(skipped):
            for pool in self.pools.values():pool.step(bgr,[],[])
        self.last_timestamp=timestamp;self.calls+=1;self.revivals=[]
        by_class={}
        for d in detections:
            if d['confidence']>=.1:by_class.setdefault(int(d['class_id']),[]).append(d)
        for cid in by_class:
            if cid not in self.pools:self.pools[cid]=ByteTracker(self.settings)
        result=[]
        for cid,pool in self.pools.items():
            current=by_class.get(cid,[])
            high_sources=[d for d in current if d['confidence']>=.35]
            low_sources=[d for d in current if .1<=d['confidence']<.35]
            high=[(np.asarray(d['bbox'],float),d['class_name'],d['confidence']) for d in high_sources]
            low=[(np.asarray(d['bbox'],float),d['class_name'],d['confidence']) for d in low_sources]
            pool.step(bgr,high,low)
            self.revivals.extend(dict(class_id=cid,old_id=a,temporary_id=b,similarity=float(s)) for a,b,s in pool.last_revivals)
            for t in pool.tracks:
                # Predictions through empty frames must never create events.
                if t.time_since_update or not t.is_displayable():continue
                tier,index=pool.last_matches[id(t)]
                source=(high_sources if tier=='high' else low_sources)[index]
                key=(cid,t.id)
                if key not in self.identities:
                    self.identities[key]=self.next_id;self.next_id+=1
                result.append(dict(source,bbox=t.last_bbox.tolist(),track_id=self.identities[key],
                                   confidence=float(t.last_score),class_id=cid,
                                   class_name=t.smoothed_label(),recovered=bool(t.recovered_last)))
        return result
