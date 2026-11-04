"""Exercise real torch tensors/checkpoint loading with a stub official forward.
This verifies our adapter, not SANSA model quality or real-checkpoint compatibility.
"""
from pathlib import Path
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from triffid_uav_perception.models.sansa import SansaBackend
from triffid_uav_perception.ontology import classes_for


class OfficialForwardStub(torch.nn.Module):
    def __init__(self, shots):
        super().__init__()
        self.adapter = torch.nn.Linear(1, 1)
        self.shots = shots
        self.calls = []

    def forward(self, video, prompt):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert video.shape == (1, self.shots + 1, 3, 16, 16)
        assert prompt['shots'] == self.shots
        for i in range(self.shots):
            entry = prompt[0][i]
            assert entry['prompt_type'] == 'mask'
            assert entry['prompt'].shape == (1, 1, 16, 16)
            assert set(entry['prompt'].unique().tolist()) <= {0, 1}
        self.calls.append((video.clone(), prompt))
        logits = torch.full((self.shots + 1, 4, 4), -10.)
        logits[-1, 1:3, 1:3] = 10.
        return {'pred_masks': logits}


@pytest.mark.parametrize('shots', [1, 5])
def test_sansa_support_query_contract(tmp_path, monkeypatch, shots, bank_selection):
    repo = tmp_path / 'SANSA'
    source = repo / 'models/sansa/sansa.py'
    source.parent.mkdir(parents=True)
    source.write_text('# Stub used only by adapter tests\n')
    base = tmp_path / 'sam2.pt'
    base.write_bytes(b'test base checkpoint')
    model = OfficialForwardStub(shots)
    checkpoint = tmp_path / 'universal.pth'
    torch.save({'model': model.state_dict()}, checkpoint)
    paths = types.ModuleType('util.path_utils')
    paths.SAM2_PATHS_CONFIG = {'large': ('old.pt', 'config.yaml')}
    builder = types.ModuleType('models.sansa.sansa')
    builder.__file__ = str(source)
    builds = []
    def build_sansa(*args):
        assert args == ('large', [2, 3], .8, 'cpu')
        assert paths.SAM2_PATHS_CONFIG['large'][0] == str(base)
        assert Path.cwd() == repo
        builds.append(args)
        return model
    builder.build_sansa = build_sansa
    monkeypatch.setitem(sys.modules, 'util.path_utils', paths)
    monkeypatch.setitem(sys.modules, 'models.sansa.sansa', builder)
    old_path = sys.path[:]
    original_cwd = Path.cwd()
    try:
        backend = SansaBackend(repo, checkpoint, base, device='cpu', image_size=16)
        classes = classes_for('Point', 'debris,person')
        supports = bank_selection(classes, shots)
        backend.prepare(classes, supports)
        assert Path.cwd() == original_cwd
        assert paths.SAM2_PATHS_CONFIG['large'] == ('old.pt', 'config.yaml')
        assert len(builds) == 1
        assert not any(p.requires_grad for p in model.parameters())
        predictions = list(backend.predict(np.full((48, 64, 3), 255, np.uint8), classes))
        assert len(predictions) == 2 and len(model.calls) == 2
        assert predictions[0].mask.shape == (48, 64)
        assert predictions[0].mask.dtype == np.bool_
        assert predictions[0].mask[24, 32]
        assert not predictions[0].mask[0, 0]
        assert model.calls[0][0][0, -1, 0, 0, 0].item() == pytest.approx((1 - .485) / .229)
        assert backend.metadata()['supports_per_class'] == {'16': shots, '23': shots}
    finally:
        sys.path[:] = old_path
