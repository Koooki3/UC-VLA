import sys, yaml
from collections import Counter, defaultdict
from ucvla.sodh import SODHConfig, generate_suite
from ucvla.splits import assign_splits, SPLITS
from ucvla.adapters import SODHAdapter

CFG_PATH = "/home/kooki3/UC-VLA/configs/sodh/sodh_v1.yaml"

def load(cfg_over=None):
    raw = yaml.safe_load(open(CFG_PATH))
    if cfg_over: raw.update(cfg_over)
    cfg = SODHConfig.from_dict(raw)
    suite = generate_suite(cfg)
    asg = assign_splits(suite.pairs, cfg)
    return cfg, suite, asg

def lab(e): return e.target_for(e.queries[0].query_id)
