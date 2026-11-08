"""Conservative identity/class decoupling; frozen MASA features, no learned weights."""
from dataclasses import dataclass
import math
import numpy as np


def ensure_masa_hooks(path):
    """Install optional hooks against the pinned upstream source, idempotently.

    Keep the integration reproducible after re-downloading the ignored vendor
    checkout. Unexpected versions fail rather than silently ignoring the policy.
    """
    source=path.read_text()
    hooks=('self.class_change_policy.prepare(', 'self.class_change_policy.apply(',
           'self.class_change_policy.filter_new(')
    if all(h in source for h in hooks):return
    if any(h in source for h in hooks):raise RuntimeError('Partial MASA class-correction patch')
    before='        bboxes, labels, scores, embeds, mask_inds = self.remove_distractor(\n'
    prepare=("        if hasattr(self, 'class_change_policy'):\n"
        '            bboxes, labels, scores, embeds, mask_inds = self.class_change_policy.prepare(\n'
        '                bboxes, labels, scores, embeds, mask_inds)\n\n')
    gate=('            if self.with_cats:\n'
        '                match_scores = match_scores.masked_fill(\n'
        '                    labels[:, None] != memo_labels[None, :], -1.0\n                )')
    apply=("            if hasattr(self, 'class_change_policy'):\n"
        '                match_scores = self.class_change_policy.apply(\n'
        '                    match_scores, match_scores_cosine, bboxes, labels,\n'
        '                    memo_bboxes, memo_labels, frame_id, memo_frame_ids)\n'
        +gate.replace('            if self.with_cats:', '            elif self.with_cats:'))
    new='        num_news = new_inds.sum()\n'
    filter_new=("        if hasattr(self, 'class_change_policy'):\n"
        '            new_inds = self.class_change_policy.filter_new(new_inds)\n')
    for needle,replacement in ((before,prepare+before),(gate,apply),(new,filter_new+new)):
        if source.count(needle)!=1:raise RuntimeError('Unexpected MASA version for class-correction hooks')
        source=source.replace(needle,replacement)
    path.write_text(source)


def pairwise_iou(a,b):
    a,b=np.asarray(a,dtype=np.float32).reshape(-1,4),np.asarray(b,dtype=np.float32).reshape(-1,4)
    size=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    intersection=size.prod(-1)
    area_a=np.maximum(0,a[:,2:]-a[:,:2]).prod(-1)
    area_b=np.maximum(0,b[:,2:]-b[:,:2]).prod(-1)
    return intersection/np.maximum(area_a[:,None]+area_b[None,:]-intersection,1e-9)


def normalized(values):
    return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)


@dataclass(frozen=True)
class AssociationSettings:
    cross_similarity: float=.85
    uncertain_similarity: float=.75
    cross_iou: float=.20
    max_center_shift: float=1.5
    max_area_ratio: float=3.
    max_gap_s: float=3.
    ambiguity_margin: float=.04
    duplicate_iou: float=.85
    duplicate_similarity: float=.95
    class_penalty: float=.02
    recovery_similarity: float=.65
    recovery_margin: float=.15


class ClassChangeAssociation:
    """Native MASA matching hooks: gate cross-class links and merge identical hypotheses."""
    def __init__(self,settings=None):
        self.settings=settings or AssociationSettings()
        self.sources={};self.evidence={};self.suppress_new=None
        self.last_diagnostics={}

    @staticmethod
    def key(box,score,label):
        return int(label),float(np.float32(score)),tuple(np.asarray(box,dtype=np.float32).tolist())

    def set_sources(self,detections):
        self.sources={}
        for d in detections:
            label=10000 if d['class_id']==-1 else d['class_id']
            evidence=d.get('role_evidence') or [dict(class_id=d['class_id'],class_name=d['class_name'],confidence=d['confidence'])]
            self.sources[self.key(d['bbox'],d['confidence'],label)]=evidence
        self.evidence={};self.suppress_new=None

    def prepare(self,bboxes,labels,scores,embeds,mask_inds):
        # Already sorted by native object score. Keep the exact representative box.
        boxes=bboxes.detach().cpu().numpy();vectors=normalized(embeds.detach().float().cpu().numpy())
        ious=pairwise_iou(boxes,boxes);cosines=vectors@vectors.T
        grouped=set();keep=[]
        for i in range(len(boxes)):
            if i in grouped:continue
            keep.append(i);members=[j for j in range(i,len(boxes)) if j not in grouped and
                ious[i,j]>=self.settings.duplicate_iou and cosines[i,j]>=self.settings.duplicate_similarity]
            grouped.update(members);votes={}
            for j in members:
                key=self.key(boxes[j],scores[j].item(),labels[j].item())
                for row in self.sources.get(key,[]):
                    cid=int(row['class_id'])
                    if cid not in votes or row['confidence']>votes[cid]['confidence']:votes[cid]=dict(row)
            self.evidence[self.key(boxes[i],scores[i].item(),labels[i].item())]=list(votes.values())
        index=labels.new_tensor(keep,dtype=labels.dtype).long()
        masks=mask_inds[index] if len(mask_inds) else mask_inds
        return bboxes[index],labels[index],scores[index],embeds[index],masks

    def apply(self,match_scores,cosine,bboxes,labels,memo_boxes,memo_labels,frame_id,memo_frames):
        s=self.settings
        boxes=bboxes.detach().cpu().numpy();previous=memo_boxes.detach().cpu().numpy()
        sims=cosine.detach().float().cpu().numpy();iou=pairwise_iou(boxes,previous)
        same=(labels[:,None]==memo_labels[None,:]).detach().cpu().numpy()
        areas=np.maximum(1.,np.prod(np.maximum(0,boxes[:,2:]-boxes[:,:2]),axis=1))
        oldareas=np.maximum(1.,np.prod(np.maximum(0,previous[:,2:]-previous[:,:2]),axis=1))
        ratios=areas[:,None]/oldareas[None,:]
        recent=(frame_id-memo_frames.detach().cpu().numpy())[None,:]<=s.max_gap_s
        centers=(boxes[:,:2]+boxes[:,2:])/2
        oldcenters=(previous[:,:2]+previous[:,2:])/2
        displacement=np.linalg.norm(centers[:,None,:]-oldcenters[None,:,:],axis=-1)
        scale=np.maximum(np.sqrt(areas[:,None]),np.sqrt(oldareas[None,:]))
        # At 1 Hz a small object can move completely beyond its previous box.
        # Permit a bounded centre displacement, still requiring distinctive ReID.
        spatial=(iou>=s.cross_iou)|(displacement/scale<=s.max_center_shift)
        geometry=spatial&(ratios>=1/s.max_area_ratio)&(ratios<=s.max_area_ratio)&recent
        possible=(sims>=s.uncertain_similarity)&geometry
        strong=(sims>=s.cross_similarity)&geometry
        # A larger ROI can change the descriptor when the viewpoint changes.
        # This recovery route needs overlap, a shorter gap and a much clearer
        # two-way appearance lead than the normal high-similarity route.
        recovery=(sims>=s.recovery_similarity)&geometry&(iou>=.05)&(displacement/scale<=1.)&(
            (frame_id-memo_frames.detach().cpu().numpy())[None,:]<=2)
        possible|=recovery
        strong|=recovery
        # Require a distinctive match in both directions. Deduplication above
        # avoids treating two labels for the very same box as two physical objects.
        distinctive=strong.copy()
        for i,j in zip(*np.where(strong&~same)):
            other_row=sims[i].copy();other_row[j]=-np.inf;other_row[~geometry[i]]=-np.inf
            other_col=sims[:,j].copy();other_col[i]=-np.inf;other_col[~geometry[:,j]]=-np.inf
            margin=s.ambiguity_margin if sims[i,j]>=s.cross_similarity else s.recovery_margin
            if sims[i,j]-other_row.max(initial=-np.inf)<margin or sims[i,j]-other_col.max(initial=-np.inf)<margin:
                distinctive[i,j]=False
        allowed=same|distinctive
        # An unresolved possible class change must not create another identity.
        self.suppress_new=np.any(possible&~same,axis=1)
        self.last_diagnostics=dict(cross_class_allowed=int(np.sum(distinctive&~same)),
                                   ambiguous_new_held=int(self.suppress_new.sum()))
        torch=__import__('torch')
        valid=torch.as_tensor(allowed,device=match_scores.device)
        cross=torch.as_tensor(distinctive&~same,device=match_scores.device)
        # Bi-softmax includes spatially impossible alternatives. Once an
        # unambiguous cross-class link passed all gates, use its ReID score.
        match_scores=torch.where(cross,torch.maximum(match_scores,cosine),match_scores)
        penalty=torch.as_tensor(~same,device=match_scores.device,dtype=match_scores.dtype)*s.class_penalty
        return (match_scores-penalty).masked_fill(~valid,-1.)

    def filter_new(self,new_inds):
        if self.suppress_new is None:return new_inds
        torch=__import__('torch')
        return new_inds&~torch.as_tensor(self.suppress_new,device=new_inds.device)


class TemporalClassEvidence:
    """Five-second bounded history keyed by persistent UUID, with hysteresis."""
    def __init__(self,window_s=5.,minimum=.35,min_support=2,margin=.15):
        self.window_ns=int(window_s*1e9);self.minimum=minimum;self.min_support=min_support;self.margin=margin

    def update(self,previous,track,timestamp_ns):
        state=dict(previous or {})
        history=[r for r in state.get('history',[]) if 0<=timestamp_ns-r['timestamp_ns']<self.window_ns]
        raw=track.get('class_evidence') or track.get('role_evidence') or [dict(
            class_id=track['class_id'],class_name=track['class_name'],confidence=track['confidence'])]
        votes={}
        for row in raw:
            score=float(row['confidence']);cid=int(row['class_id'])
            if not math.isfinite(score) or not 0<=score<=1:continue
            if score>=self.minimum and (cid not in votes or score>votes[cid]['confidence']):
                votes[cid]=dict(class_id=cid,class_name=str(row['class_name']),confidence=score)
        if votes and (not history or history[-1]['timestamp_ns']!=timestamp_ns):
            history.append(dict(timestamp_ns=timestamp_ns,votes=list(votes.values()),bbox=list(track['bbox'])))
        history=history[-5:]
        if not state:
            best=max(raw,key=lambda r:r['confidence'])
            state=dict(class_id=int(best['class_id']),class_name=best['class_name'],revision=0,
                       first_class=best['class_name'],confidence=float(best['confidence']),best=None,alert_confidence=0.)
        totals={};weights={};names={}
        for entry in history:
            decay=.8**((timestamp_ns-entry['timestamp_ns'])/1e9)
            for row in entry['votes']:
                cid=row['class_id'];names[cid]=row['class_name']
                totals[cid]=totals.get(cid,0.)+decay*row['confidence']
                weights[cid]=weights.get(cid,0.)+decay
        old_name=state['class_name'];changed=False
        if votes and totals:
            winner=max(totals,key=totals.get);total=sum(totals.values())
            recent=history[-3:]
            hits=sum(any(r['class_id']==winner for r in e['votes']) for e in recent)
            if winner!=state['class_id'] and hits>=self.min_support and (totals[winner]-totals.get(state['class_id'],0))/max(total,1e-9)>=self.margin:
                state.update(class_id=winner,class_name=names[winner],revision=min(state['revision']+1,2**31-1),best=None,alert_confidence=0.)
                changed=True
        cid=state['class_id']
        if cid in totals:state['confidence']=totals[cid]/weights[cid]
        for entry in history:
            for row in entry['votes']:
                if row['class_id']==cid and (state.get('best') is None or row['confidence']>state['best']['confidence']):
                    state['best']=dict(confidence=row['confidence'],bbox=list(entry['bbox']),timestamp_ns=entry['timestamp_ns'])
        best_score=state['best']['confidence'] if state.get('best') else state['confidence']
        improved=not changed and best_score-state.get('alert_confidence',0)>=.10
        if changed or improved:state['alert_confidence']=best_score
        state.update(history=history,last_seen_ns=timestamp_ns)
        event=dict(kind='class' if changed else 'confidence',previous_class=old_name,
                   **{'class':state['class_name']},confidence=state['confidence'],best_confidence=best_score) if changed or improved else None
        return state,event
