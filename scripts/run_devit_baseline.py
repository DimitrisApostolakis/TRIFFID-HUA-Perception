"""Compatibility entrypoint for the archived DE-ViT baseline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_fewshot_baseline import main, sample_frames

if __name__ == '__main__':
    main(default_backend='devit')
