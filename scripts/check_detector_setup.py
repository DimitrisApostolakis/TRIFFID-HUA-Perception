"""Import/checkpoint checks only. No support extraction, images, inference or timing."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(os.environ.get('TRIFFID_PROJECT_ROOT',Path(__file__).absolute().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['FSOD-VFM','DINOv','YOLOE'])
    parser.add_argument('--load-cpu',action='store_true')
    parser.add_argument('--variant',choices=['small','large'])
    args=parser.parse_args()
    args.variant=args.variant or ('small' if args.model=='DINOv' else 'large')
    import torch
    torch.set_num_threads(4)
    result=dict(model=args.model,python=sys.version.split()[0],torch=torch.__version__,
                interpreter=sys.executable,inference_executed=False,support_features_built=False,
                checkpoint_loaded_cpu=False,variant=args.variant)
    if args.load_cpu:
        def forbidden_forward(*a,**kw):
            raise RuntimeError('Model forward calls are forbidden during setup')
        torch.nn.Module._call_impl=forbidden_forward
    if args.model=='DINOv':
        from dinov import build_model
        from dinov.body.encoder.ops.functions import MSDeformAttnFunction
        from utils.arguments import load_opt_from_config_file
        result['repo']='ugv_models/dinov'
        if args.load_cpu:
            large=args.variant=='large'
            config='dinov_sam_coco_swinl_train.yaml' if large else 'dinov_sam_coco_train.yaml'
            opt=load_opt_from_config_file(str(ROOT/'ugv_models/dinov/configs'/config))
            opt['MODEL']['BACKBONE']['LOAD_PRETRAINED']=False
            opt['OUTPUT_DIR']=str(ROOT/'ugv_models/dinov_support_cache/PUI')
            # Upstream training-criterion constructors allocate a few CUDA
            # index tensors even before inference. Keep those on CPU during
            # this CPU-only structural check; production source is unchanged.
            native_cuda=torch.Tensor.cuda
            torch.Tensor.cuda=lambda tensor,*a,**kw:tensor
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    model=build_model(opt)
            finally:
                torch.Tensor.cuda=native_cuda
            path=ROOT/'ugv_models/weights/DINOv'/('model_swinL.pth' if large else 'model_swinT.pth')
            state=torch.load(path,map_location='cpu')
            state=state.get('model',state)
            state={k[6:]:v for k,v in state.items() if k.startswith('model.')}
            # The released checkpoint retains an obsolete loss-only buffer;
            # it is not a detector parameter. Keep strict loading for all
            # actual architecture weights.
            ignored=[]
            if 'criterion.empty_weight' in state and 'criterion.empty_weight' not in model.state_dict():
                state.pop('criterion.empty_weight')
                ignored.append('criterion.empty_weight')
            model.load_state_dict(state,strict=True)
            result['ignored_training_only_keys']=ignored
    elif args.model=='FSOD-VFM':
        import model.dinov2, model.sam2, support_util, query_util, metric
        from chatrex.upn import UPNWrapper
        from chatrex.upn import build_architecture
        from mmengine.config import Config
        from sam2.build_sam import build_sam2
        from sam2 import _C
        result['repo']='ugv_models/fsod_vfm'
        if args.load_cpu:
            # Construct and load on CPU; UPNWrapper itself calls .cuda(), so
            # validate its underlying builder instead.
            cfg=Config.fromfile(str(ROOT/'ugv_models/fsod_vfm/chatrex/upn/configs/upn_large.py')).model
            model=build_architecture(cfg)
            state=torch.load(ROOT/'ugv_models/weights/FSOD-VFM/upn_large.pth',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True)
            del state
            del model
            model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
                str(ROOT/'ugv_models/weights/FSOD-VFM/sam2.1_hiera_large.pt'),device='cpu')
            del model
            model=torch.hub.load(str(ROOT/'ugv_models/fsod_vfm/dinov2'),
                                 'dinov2_vitl14',source='local',pretrained=False)
            state=torch.load(ROOT/'ugv_models/fsod_vfm/checkpoints/dinov2_vitl14_pretrain.pth',
                             map_location='cpu',weights_only=False)
            model.load_state_dict(state,strict=True)
    else:
        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
        result['repo']='ugv_models/yoloe'
        if args.load_cpu:
            name='yoloe-v8l-seg.pt' if args.variant=='large' else 'yoloe-v8s-seg.pt'
            model=YOLOE(str(ROOT/'ugv_models/weights/YOLOE'/name))
            assert hasattr(model.model.model[-1],'savpe'),'Checkpoint has no visual prompt encoder'
    result['checkpoint_loaded_cpu']=args.load_cpu
    result['source_revision']=subprocess.check_output(
        ['git','-C',str(ROOT/result['repo']),'rev-parse','HEAD'],text=True).strip()
    destination=ROOT/'ugv_models/setup_status'
    destination.mkdir(exist_ok=True)
    (destination/f'{args.model}_{args.variant}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
