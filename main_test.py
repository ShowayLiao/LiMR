#!/usr/bin/python3

import logging
import os
import pprint


from tools.train import train_model,train
from tools.test import test_model
from utils import setup_logging, load_config, parse_args,set_output_dir,get_max_epoch_pth_file

from tools import train,test
import time

LOGGER = logging.getLogger(__name__)

def main():

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config=path_to_config)

        # set output directory and logging
        cfg = set_output_dir(cfg)
        setup_logging(cfg)
        LOGGER.info(pprint.pformat(cfg))
        test_(cfg, path_to_config)


def test_(cfg, path_to_config):
    LOGGER.info("time is {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    LOGGER.info(pprint.pformat(cfg))
    LOGGER.info("path_to_config is {}".format(path_to_config))

    LOGGER.info("start testing!")
    """
    include:
     1) test dataloader load
     2) test prepare phase: 1) base model load
                            2) start test (one follow by one)
    """
    test_model(cfg=cfg)

    LOGGER.info("Testing complete!")









# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
