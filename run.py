import hydra
from hydra.experimental import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os
from utils.vis import Logger
import sys
import logging
from gabor2d.train import gaborlpinn_ent

def setup_logging():
    """Enable pretty logging and sets the level to DEBUG."""
    logging.addLevelName(logging.DEBUG, "D")
    logging.addLevelName(logging.INFO, "I")
    logging.addLevelName(logging.WARNING, "W")
    logging.addLevelName(logging.ERROR, "E")
    logging.addLevelName(logging.CRITICAL, "C")

    formatter = logging.Formatter(
        fmt=("%(levelname)s%(asctime)s" " [%(module)s:%(lineno)d] %(message)s"),
        datefmt="%m%d %H:%M:%S",
    )

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(console_handler)

    return logger


def init(cfg):
    setup_logging()
    pass

def exit(cfg):
    pass

def print_log(cfg):
    if not os.path.exists(cfg.results_path):
        os.makedirs(cfg.results_path)
    sys.stdout = Logger(os.path.join(cfg.results_path,'out.txt'))
    print(OmegaConf.to_yaml(cfg))

def run_function(cfg):
    task = cfg.task
    if task == 'gaborlpinn':
        gaborlpinn_ent(cfg)

@hydra.main(config_path="conf", config_name="config")
def run(cfg:DictConfig):
    print_log(cfg)
    init(cfg)
    run_function(cfg)
    exit(cfg)

def run_debug():
    # global initialization
    with initialize(config_path="conf", job_name="test_app"):
        cfg = compose(config_name="config")
        print_log(cfg)
        init(cfg)
        run_function(cfg)
        exit(cfg)

if __name__ == "__main__":
    run()