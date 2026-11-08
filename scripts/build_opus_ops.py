"""Build pinned MMCV against OPUS's torch and repository-local CUDA toolkit."""
import os
from pathlib import Path
import subprocess
import sys

import torch


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(CUDA_HOME=str(root / 'ugv_models/opus_cuda'), FORCE_CUDA='1', MMCV_WITH_OPS='1',
               TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS=env.get('OPUS_BUILD_JOBS', '4'), CC='/usr/bin/gcc', CXX='/usr/bin/g++')
    # PyTorch's pip CUDA libraries already contain their matching public headers.
    nvidia = Path(torch.__file__).resolve().parents[1] / 'nvidia'
    include_dirs = [str(root / 'ugv_models/opus_cuda/targets/x86_64-linux/include')]
    include_dirs += [str(path) for path in sorted(nvidia.glob('*/include'))]
    env['CPATH'] = ':'.join(include_dirs + ([env['CPATH']] if env.get('CPATH') else []))
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation',
                    '--no-deps', '--no-clean', str(root / 'ugv_models/opus/third_party/mmcv'),
                    '--log', '/tmp/opus_mmcv_build_final.log'], env=env, check=True)
