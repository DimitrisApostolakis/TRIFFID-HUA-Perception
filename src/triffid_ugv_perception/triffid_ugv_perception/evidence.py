"""Bounded ROS-free transport for appearance and image/world association evidence."""
import json
import math
import numpy as np
from .ontology import BY_ID

SCHEMA='triffid.ugv.tracked-observations.v1'

def validate_packet(packet,max_tracks=512,max_dims=512):
    if not isinstance(packet,dict) or packet.get('schema')!=SCHEMA:
        raise ValueError('Unknown tracked-observation schema')
    stamp=packet.get('timestamp_ns')
    if type(stamp) is not int or not 0<=stamp<2**63:raise ValueError('Invalid observation timestamp')
    for key in ('frame_id','embedding_space','session_id'):
        if not isinstance(packet.get(key),str) or not 0<len(packet[key])<=128:
            raise ValueError(f'Invalid {key}')
    if not packet['embedding_space'].startswith('MASA-R50:'):raise ValueError('Expected MASA embedding space')
    tracks=packet.get('tracks')
    if not isinstance(tracks,list) or len(tracks)>max_tracks:raise ValueError('Oversized track frame')
    ids=set()
    for t in tracks:
        tid=t.get('track_id')
        if not isinstance(tid,str) or len(tid)>128 or not tid.startswith(packet['session_id']+':') or tid in ids:
            raise ValueError('Invalid or repeated session-scoped track ID')
        ids.add(tid)
        c=BY_ID.get(t.get('class_id'))
        if c is None or c.name!=t.get('class_name'):raise ValueError('Invalid observation class')
        score=t.get('confidence')
        if not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Invalid observation confidence')
        for key,n in [('bbox',4),('position',3),('extent',3)]:
            a=np.asarray(t.get(key),dtype=float)
            if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f'Invalid {key}')
        if t['bbox'][2]<=t['bbox'][0] or t['bbox'][3]<=t['bbox'][1] or min(t['extent'])<0:
            raise ValueError('Invalid image box or physical extent')
        a=np.asarray(t.get('appearance'),dtype=float)
        if a.ndim!=1 or not 0<len(a)<=max_dims or not np.isfinite(a).all() or np.linalg.norm(a)<=0:
            raise ValueError('Invalid appearance descriptor')
    return packet

def encode_packet(tracks,timestamp_ns,frame_id,embedding_space,session_id,max_bytes=2*1024*1024):
    keys=('track_id','class_id','class_name','confidence','bbox','appearance','position','extent',
          'identity_class_id','recovered','role_ambiguous','role_share')
    rows=[{k:t[k] for k in keys if k in t} for t in tracks]
    packet=dict(schema=SCHEMA,timestamp_ns=int(timestamp_ns),frame_id=frame_id,
        embedding_space=embedding_space,session_id=session_id,tracks=rows)
    validate_packet(packet)
    encoded=json.dumps(packet,separators=(',',':'),allow_nan=False)
    if len(encoded.encode())>max_bytes:raise ValueError('Observation packet exceeds byte limit')
    return encoded

def decode_packet(text,max_bytes=2*1024*1024):
    if not isinstance(text,str) or len(text.encode())>max_bytes:raise ValueError('Observation packet exceeds byte limit')
    return validate_packet(json.loads(text))
