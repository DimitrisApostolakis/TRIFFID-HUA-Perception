"""Run COSINE with the shared offline evaluation and persistent registry."""
from run_fewshot_baseline import main

if __name__=='__main__':
    main(default_backend='cosine')
