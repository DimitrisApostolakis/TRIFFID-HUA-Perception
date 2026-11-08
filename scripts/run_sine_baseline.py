"""Run local pretrained SINE with the shared 1 Hz tracking/event baseline."""
from run_fewshot_baseline import main

if __name__ == '__main__':
    main(default_backend='sine')
