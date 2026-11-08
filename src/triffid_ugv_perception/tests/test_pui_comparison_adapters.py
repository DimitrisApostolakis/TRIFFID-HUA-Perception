"""Regression for a thin annotated cone disappearing on FSOD's feature grid."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path('/mnt/d/projects/triffid-hua-perception/scripts/pui_comparison_detectors.py')
spec=importlib.util.spec_from_file_location('pui_comparison_adapters',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_area_pooling_preserves_tiny_annotated_foreground():
    mask=np.zeros((64,64),np.uint8);mask[5:7,5:7]=1
    weights=module.area_mask_weights(mask,(4,4))
    assert weights.shape==(4,4) and weights.dtype==np.float32
    assert weights.sum()>0 and np.isfinite(weights).all()
    # Four genuine source pixels occupy 4/256 of one feature cell; no filled box.
    np.testing.assert_allclose(weights.sum(),4/256)
    assert np.count_nonzero(weights)==1


def test_failed_model_is_audited_and_returns_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import subprocess
    import json
    script=path.parent/'run_pui_comparison.py'
    runner_spec=importlib.util.spec_from_file_location('pui_runner_test',script)
    runner=importlib.util.module_from_spec(runner_spec);runner_spec.loader.exec_module(runner)
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'ugv_models').mkdir()
    (tmp_path/'ugv_supports/PUI').mkdir(parents=True)
    (tmp_path/'ugv_supports/PUI/bank.json').write_text('{}')
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'frames.jsonl').write_text('{}\n')
    def failure(command,**kwargs):raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(runner.subprocess,'run',failure)
    args=SimpleNamespace(models=['YOLOE'],dataset=dataset,run='failure-test',force=False,limit=0)
    assert runner.run(args)==1
    audit=json.loads((tmp_path/'samples/YOLOE/original/failure-test/audit.json').read_text())
    assert audit['status']=='failed' and audit['api_enabled'] is False
