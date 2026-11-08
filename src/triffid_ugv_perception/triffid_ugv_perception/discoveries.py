"""Session object registry and idempotent local creation events.

Stable track IDs are the primary association. After track loss, a conservative
same-class appearance + image-overlap check can rebind an object. RGB-only
reidentification is not a guarantee after arbitrary camera/object motion.
"""
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from .io import write_json
from .tracking.metrics import _iou_batch


def cosine(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape != b.shape or a.size == 0:
        return -1.
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator>0 else -1.


class ObjectRegistry:
    def __init__(self, path, confirm_frames=2, reid_similarity=.90, reid_iou=.15, context=None):
        if confirm_frames<1 or not 0<=reid_similarity<=1 or not 0<=reid_iou<=1:
            raise ValueError('Invalid registry confirmation/association settings')
        self.path=Path(path)
        self.confirm_frames=confirm_frames
        self.reid_similarity=reid_similarity
        self.reid_iou=reid_iou
        self.context=context
        self.objects={}
        if self.path.exists():
            data=json.loads(self.path.read_text())
            if data.get('schema')!='triffid.discoveries.v1':
                raise ValueError('Unknown object registry schema')
            if data.get('context')!=context:
                raise ValueError('Registry context differs: use the same experiment/camera/appearance model or a separate registry')
            self.objects=data['objects']
        # Tracker IDs may restart at 1 in a new process. Never reuse saved aliases.
        self.aliases={}
        self.session=uuid4().hex
        self.candidates={}
        self.frame=0

    def update(self, tracks, timestamp_ns):
        self.frame+=1
        identities=[(int(t.get('identity_class_id',t['class_id'])),int(t['track_id'])) for t in tracks]
        if len(set(identities))!=len(identities):
            raise ValueError('Repeated track identity in one frame')
        used={self.aliases[k] for k in identities if k in self.aliases}
        created=[]
        observations=[]
        for track,key in zip(tracks,identities):
            candidate=self.candidates.get(key)
            hits=candidate['hits']+1 if candidate and candidate['frame']==self.frame-1 else 1
            self.candidates[key]=dict(frame=self.frame,hits=hits)
            object_id=self.aliases.get(key)
            if object_id is None:
                matches=[]
                for oid,known in self.objects.items():
                    if oid in used or known.get('identity_class_id',known['class_id'])!=key[0]:
                        continue
                    overlap=float(_iou_batch([known['bbox']],[track['bbox']])[0,0])
                    similarity=cosine(known.get('appearance',[]),track.get('appearance',[]))
                    if overlap>=self.reid_iou and similarity>=self.reid_similarity:
                        matches.append((similarity+overlap,oid))
                if matches:
                    object_id=max(matches)[1]
                elif hits>=self.confirm_frames:
                    object_id=uuid4().hex
                    event=dict(type='object_created',object_id=object_id,class_id=int(track['class_id']),
                        class_name=track['class_name'],timestamp_ns=int(timestamp_ns),
                        bbox=list(map(float,track['bbox'])),confidence=float(track['confidence']),
                        track_id=int(track['track_id']),session_id=self.session)
                    self.objects[object_id]=dict(class_id=int(track['class_id']),identity_class_id=key[0],created_event=event)
                    created.append(event)
                if object_id is not None:
                    # A saved object has only one current tracker alias. A
                    # reappearing stale alias must not assign its UUID twice.
                    self.aliases={alias:oid for alias,oid in self.aliases.items()
                                  if oid!=object_id}
                    self.aliases[key]=object_id
                    used.add(object_id)
            if object_id is not None:
                known=self.objects[object_id]
                known.update(class_id=int(track['class_id']),class_name=track['class_name'],
                    bbox=list(map(float,track['bbox'])),last_seen_ns=int(timestamp_ns),
                    appearance=list(map(float,track.get('appearance',known.get('appearance',[])))))
            observations.append(dict(track,object_id=object_id))
        # Do not retain an unbounded history of unconfirmed ephemeral tracks.
        self.candidates={k:v for k,v in self.candidates.items() if v['frame']>=self.frame-1}
        write_json(self.path,dict(schema='triffid.discoveries.v1',context=self.context,objects=self.objects))
        return observations,created

    def write_creation_events(self, directory):
        """One atomic file per stable UUID; recover pending files after a crash.

        This local outbox does not claim exactly-once remote delivery. It is not
        connected to the API. A future sender must track acknowledgements.
        """
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for oid,known in self.objects.items():
            path=directory/(oid+'.json')
            if not path.exists():
                write_json(path,known['created_event'])
