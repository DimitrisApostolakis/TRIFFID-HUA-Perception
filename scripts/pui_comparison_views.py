"""Separate visual stages and unlabelled competing-class diagnostics. No inference/network."""
from itertools import combinations
from pathlib import Path
import json

import cv2
import numpy as np

PERSON_ROLES=frozenset(('citizen','civilian','first responder','military personnel','person'))
VIDEO_NAMES=('detections','tracking','exports')
VIEW_SCHEMA=3


def overlap(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]));y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area=x*y
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union>0 else 0.


def competing_labels(detections,tracks,threshold=.5):
    """Diagnostic overlap, not a GT confusion matrix or a classification error rate."""
    pairs=[]
    for i,j in combinations(range(len(detections)),2):
        a,b=detections[i],detections[j]
        if a['class_name']==b['class_name']:continue
        iou=overlap(a['bbox'],b['bbox'])
        if iou<threshold:continue
        pair=sorted((a['class_name'],b['class_name']))
        pairs.append(dict(classes=pair,iou=round(iou,4),indices=[i,j],
                          person_roles=all(name in PERSON_ROLES for name in pair)))
    disagreements=[]
    for i,raw in enumerate(detections):
        if raw['class_name'] not in PERSON_ROLES:continue
        for current in tracks:
            if (current['class_name'] in PERSON_ROLES and current['class_name']!=raw['class_name']
                    and overlap(raw['bbox'],current['bbox'])>=threshold):
                disagreements.append(dict(raw_index=i,raw_class=raw['class_name'],
                    track_class=current['class_name'],track_id=current['track_id']))
    return dict(iou_threshold=threshold,raw_cross_class_pairs=pairs,raw_track_role_disagreements=disagreements)


def new_ready_events(queued_ids,seen):
    """Observe actual queue admission without acquiring a delivery lease or sending."""
    fresh=set(queued_ids)-seen
    seen.update(fresh)
    return sorted(fresh)


def rows_for_view(packet,view):
    if view=='detections':return packet['detections']
    if view=='tracking':return packet['tracks']
    if view=='exports':
        selected=set(packet['new_object_ids'])|set(packet.get('updated_object_ids',[]))
        rows=[row for row in packet['tracks'] if row.get('object_id') in selected]
        if len({row['object_id'] for row in rows})!=len(rows) or {row['object_id'] for row in rows}!=selected:
            raise ValueError('Each new GeoJSON UUID must have exactly one current observation')
        return rows
    raise ValueError('Unknown comparison view')


def geojson_collection(packets,model):
    """One first-confirmation feature per UUID; image coordinates are properties only."""
    if any(p.get('class_correction') for p in packets):
        return corrected_geojson_collection(packets,model)
    seen=set();features=[]
    for packet in packets:
        for row in rows_for_view(packet,'exports'):
            oid=row['object_id']
            if oid in seen:raise ValueError('Repeated GeoJSON discovery UUID')
            seen.add(oid)
            features.append(dict(type='Feature',id=oid,geometry=None,properties=dict(
                object_id=oid,track_id=row['track_id'],**{'class':row['class_name']},
                class_id=row['class_id'],confidence=row['confidence'],bbox_px=list(row['bbox']),
                discovered_frame=packet['frame'],timestamp_ns=packet['source']['bag_timestamp_ns'],
                image=packet['source']['file'],source='ugv',location_valid=False,
                snapshot='first_confirmed_observation')))
    return dict(type='FeatureCollection',features=features,metadata=dict(model=model,
        policy='One feature per newly confirmed registry UUID; repeated sightings excluded',
        geometry_note='World coordinates unavailable; bbox_px uses image pixels and geometry is null',
        api_enabled=False))


def corrected_geojson_collection(packets,model):
    """Upsert local features; identity/discovery stay fixed as evidence improves."""
    features={};sources={p['source']['bag_timestamp_ns']:p['source'] for p in packets}
    for packet in packets:
        rows_for_view(packet,'exports')  # Reject missing or duplicate event observations.
        for oid in packet['new_object_ids']:
            if oid in features:raise ValueError('Repeated GeoJSON discovery UUID')
            features[oid]=dict(type='Feature',id=oid,geometry=None,properties=dict(
                object_id=oid,discovered_frame=packet['frame'],
                discovered_timestamp_ns=packet['source']['bag_timestamp_ns'],
                source='ugv',location_valid=False))
        if not set(packet.get('updated_object_ids',[]))<=features.keys():
            raise ValueError('Cannot update an undiscovered GeoJSON UUID')
        for row in packet['tracks']:
            oid=row.get('object_id')
            if oid not in features:continue
            best=row.get('best_observation') or dict(confidence=row['confidence'],
                bbox=row['bbox'],timestamp_ns=packet['source']['bag_timestamp_ns'])
            props=features[oid]['properties']
            props.update(track_id=row['track_id'],**{'class':row['class_name']},
                class_id=row['class_id'],confidence=best['confidence'],
                class_confidence=row['confidence'],bbox_px=list(best['bbox']),
                timestamp_ns=best['timestamp_ns'],image=sources[best['timestamp_ns']]['file'],
                last_seen_ns=packet['source']['bag_timestamp_ns'],latest_bbox_px=list(row['bbox']),
                first_class=row.get('first_class',row['class_name']),
                class_revision=row.get('class_revision',0),snapshot='best_observation_of_stable_class')
    return dict(type='FeatureCollection',features=list(features.values()),metadata=dict(model=model,
        policy='One feature per confirmed UUID; local class/confidence updates replace that feature',
        geometry_note='World coordinates unavailable; bbox_px uses image pixels and geometry is null',
        api_enabled=False))


def color_for(identity):
    return (int(60+identity*71%195),int(60+identity*113%195),int(60+identity*43%195))


def label(canvas,text,point,scale=.45):
    h,w=canvas.shape[:2]
    width=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0]
    x=max(3,min(int(point[0]),max(3,w-width-4)));y=max(73,min(int(point[1]),h-7))
    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)


def render(image,packet,view,model,total,mask_base=None):
    # Masks belong only to the accepted-detection view, never to tracked/export views.
    canvas=(mask_base if view=='detections' and mask_base is not None else image).copy()
    rows=rows_for_view(packet,view)
    for row in rows:
        identity=int(row.get('track_id',row['class_id']))
        color=color_for(identity)
        a,b,c,d=map(int,row['bbox']);cv2.rectangle(canvas,(a,b),(c,d),color,2)
        text=f"{row['class_name']} {row['confidence']:.0%}"
        if packet.get('class_correction') and view=='tracking':
            text+=f" (obs {row.get('observation_confidence',row['confidence']):.0%})"
        if packet.get('class_correction') and view=='exports' and row.get('best_observation'):
            text=f"{row['class_name']} best {row['best_observation']['confidence']:.0%}"
        if view!='detections':
            text+=f" T{row['track_id']}"
            if row.get('object_id'):text+=f" {row['object_id'][:6]}"
            if view=='tracking' and row.get('role_ambiguous'):text+=' role uncertain'
            if view=='tracking' and row.get('recovered',row.get('confidence',1)<.35):text+=' recovered'
        if view=='exports':text+=' NEW GEOJSON' if row['object_id'] in packet['new_object_ids'] else ' UPDATED'
        label(canvas,text,(a,b-6))
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],58),(20,20,20),-1)
    title_name='GEOJSON EXPORTS' if view=='exports' else view.upper()
    title=f"{model} | {title_name} | {packet['frame']+1}/{total} | 1 FPS input"
    cv2.putText(canvas,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1,cv2.LINE_AA)
    if view=='detections':detail=f'{len(rows)} accepted predictions | threshold 0.35 | no tracks'
    elif view=='tracking':detail=f'{len(rows)} MASA tracks | one label per track | no raw prediction layer'
    else:detail=f"{len(packet['new_object_ids'])} new / {len(packet.get('updated_object_ids',[]))} updated | geometry: null | API OFF | 0 sent"
    cv2.putText(canvas,detail,(12,47),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1,cv2.LINE_AA)
    if view=='exports' and not rows:
        label(canvas,'No new objects or significant updates in this frame.',(12,canvas.shape[0]-16),.49)
    return canvas


class VideoSet:
    def __init__(self,directory,model,total):
        self.directory=Path(directory);self.model=model;self.total=total;self.writers={};self.count=0
        for name in VIDEO_NAMES:(self.directory/'overlays'/name).mkdir(parents=True,exist_ok=True)

    def write(self,image,packet,mask_base=None):
        for name in VIDEO_NAMES:
            canvas=render(image,packet,name,self.model,self.total,mask_base)
            if name not in self.writers:
                writer=cv2.VideoWriter(str(self.directory/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1.,
                                       (image.shape[1],image.shape[0]))
                if not writer.isOpened():raise OSError(f'Cannot open {name} video')
                self.writers[name]=writer
            self.writers[name].write(canvas)
            if not cv2.imwrite(str(self.directory/'overlays'/name/f"{packet['frame']:04d}.jpg"),canvas):
                raise OSError(f'Cannot write {name} overlay')
        self.count+=1

    def close(self):
        for writer in self.writers.values():writer.release()


def summarize_packets(packets):
    all_pairs={};role_pairs={};raw_disagreement_frames=[];role_overlap_frames=[]
    for packet in packets:
        diagnostic=packet['competing_labels']
        if diagnostic['raw_track_role_disagreements']:raw_disagreement_frames.append(packet['frame'])
        person_found=False
        for pair in diagnostic['raw_cross_class_pairs']:
            key=' / '.join(pair['classes'])
            all_pairs.setdefault(key,set()).add(packet['frame'])
            if pair['person_roles']:
                role_pairs.setdefault(key,set()).add(packet['frame']);person_found=True
        if person_found:role_overlap_frames.append(packet['frame'])
    return dict(diagnostic_only=True,ground_truth_evaluated=False,iou_threshold=.5,
        raw_cross_class_overlap_frames={k:sorted(v) for k,v in sorted(all_pairs.items())},
        raw_person_role_overlap_frames={k:sorted(v) for k,v in sorted(role_pairs.items())},
        raw_person_role_overlap_frame_count=len(role_overlap_frames),
        raw_track_role_disagreement_frames=raw_disagreement_frames,
        note='Overlap can involve separate nearby objects or legitimate overlapping semantic regions. Raw-versus-track disagreements also reflect five-second role smoothing. These are not GT classification errors.')


def install_replacement(staged,output,expected_frames):
    """Verify complete replacements first; then remove only the obsolete combined outputs."""
    staged,output=Path(staged),Path(output)
    for name in VIDEO_NAMES:
        capture=cv2.VideoCapture(str(staged/f'{name}.mp4'))
        if not capture.isOpened():raise OSError(f'Missing staged {name} video')
        decoded=0
        while True:
            ok,image=capture.read()
            if not ok:break
            decoded+=1
        capture.release()
        if decoded!=expected_frames:raise ValueError(f'Incomplete {name} video: {decoded}/{expected_frames}')
        if len(list((staged/'overlays'/name).glob('*.jpg')))!=expected_frames:
            raise ValueError('Incomplete staged overlay set')
    # Every old video remains intact until all three replacements have decoded.
    import os
    output.mkdir(parents=True,exist_ok=True)
    for name in VIDEO_NAMES:
        destination=output/'overlays'/name
        destination.mkdir(parents=True,exist_ok=True)
        for previous in destination.glob('*.jpg'):previous.unlink()
        for source in (staged/'overlays'/name).glob('*.jpg'):os.replace(source,destination/source.name)
        os.replace(staged/f'{name}.mp4',output/f'{name}.mp4')
    for name in ('timings.csv','observations.jsonl','detections.geojson','audit.json'):
        os.replace(staged/name,output/name)
    previous=output/'video.mp4'
    if previous.exists():previous.unlink()
    for previous in (output/'overlays').glob('*.jpg'):previous.unlink()
