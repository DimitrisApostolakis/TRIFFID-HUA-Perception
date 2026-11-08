"""Tracking selection and optional physical-person / role separation."""
from collections import defaultdict, deque
from .vendor.udc_core import iou_xyxy

PERSON_ROLES=frozenset({'citizen','first responder','military personnel'})
PERSON_ID=-1  # Internal identity category; never an exported ontology ID.


def merge_person_hypotheses(detections,iou_threshold=.7):
    """Merge heavily overlapping role alternatives; keep other classes intact."""
    output=[dict(d) for d in detections if d['class_name'] not in PERSON_ROLES]
    people=sorted((d for d in detections if d['class_name'] in PERSON_ROLES),key=lambda d:d['confidence'],reverse=True)
    while people:
        best=people.pop(0);cluster=[best];remaining=[]
        for d in people:
            if iou_xyxy(best['bbox'],d['bbox'])>=iou_threshold:cluster.append(d)
            else:remaining.append(d)
        people=remaining;evidence={}
        for d in cluster:
            cid=int(d['class_id'])
            if cid not in evidence or d['confidence']>evidence[cid]['confidence']:
                evidence[cid]=dict(class_id=cid,class_name=d['class_name'],confidence=float(d['confidence']))
        output.append(dict(best,class_id=PERSON_ID,class_name='person',role_evidence=list(evidence.values())))
    return output


class TrackingPipeline:
    def __init__(self,tracker='botsort',person_roles=False,udc_proof_threshold=.35,class_correction=False,masa_repo=None,masa_checkpoint=None,memory_seconds=60):
        self.class_correction=class_correction
        if class_correction and tracker!='masa':raise ValueError('Class correction currently requires MASA')
        if tracker=='masa':
            from .masa import MasaTracker
            self.engine=MasaTracker(class_correction=class_correction,repo=masa_repo,checkpoint=masa_checkpoint,memory_seconds=memory_seconds)
        elif tracker=='udc':
            from .udc import UdcTracker
            self.engine=UdcTracker(proof_threshold=udc_proof_threshold)
        elif tracker=='botsort':
            from .botsort import BotSortTracker
            self.engine=BotSortTracker()
        else:raise ValueError('Unknown tracker')
        self.name=tracker;self.person_roles=person_roles;self.history=defaultdict(deque)
        self.calls=0;self.last_timestamp=None

    def metadata(self):
        return dict(name=self.name,class_correction=self.class_correction,person_roles=self.person_roles,person_merge_iou=.7,
                    role_window_seconds=5,engine=self.engine.metadata() if hasattr(self.engine,'metadata') else 'existing BoT-SORT settings')

    def update(self,detections,bgr,timestamp_ns=None):
        timestamp=int(timestamp_ns) if timestamp_ns is not None else self.calls*10**9
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:raise ValueError('Tracking timestamps must increase')
        self.last_timestamp=timestamp
        inputs=merge_person_hypotheses(detections) if self.person_roles else detections
        tracks=self.engine.update(inputs,bgr,timestamp_ns=timestamp)
        self.calls+=1
        if self.class_correction:
            for t in tracks:
                evidence=t.get('class_evidence') or t.get('role_evidence')
                if evidence:
                    best=max(evidence,key=lambda r:r['confidence'])
                    t.update(class_id=best['class_id'],class_name=best['class_name'])
                t['identity_class_id']=-2
            return tracks
        for t in tracks:
            if t['class_id']!=PERSON_ID:continue
            # Both adapters retain the detection selected by association. A new
            # nearest-box lookup can steal a neighbour's role after filtering.
            source_evidence=t['role_evidence']
            history=self.history[t['track_id']]
            history.append((timestamp,source_evidence))
            while history and timestamp-history[0][0]>=5e9:history.popleft()
            evidence={}
            for _,rows in history:
                for row in rows:
                    cid=row['class_id']
                    evidence.setdefault(cid,dict(row,score_sum=0.))['score_sum']+=row['confidence']
            ranked=sorted(evidence.values(),key=lambda row:row['score_sum'],reverse=True)
            best=ranked[0];total=sum(row['score_sum'] for row in ranked)
            t.update(class_id=best['class_id'],class_name=best['class_name'],identity_class_id=PERSON_ID,
                     role_evidence=source_evidence,role_share=best['score_sum']/total,
                     role_ambiguous=len(ranked)>1 and (best['score_sum']-ranked[1]['score_sum'])/total<.2)
        # Keep role state through the 60-second gallery, bounded in time.
        self.history=defaultdict(deque,{tid:h for tid,h in self.history.items() if h and timestamp-h[-1][0]<=75e9})
        return tracks
