#!/usr/bin/python3

import logging
import os
import pprint

from utils import setup_logging, load_config, parse_args,set_output_dir

from tools import train


LOGGER = logging.getLogger(__name__)

def main():


    # 读取config文件

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    for path_to_config in args.cfg_files:

        cfg = load_config(args, path_to_config=path_to_config)
        cfg = set_output_dir(cfg)
        setup_logging(cfg)
        LOGGER.info(pprint.pformat(cfg))
        train(cfg=cfg)
    LOGGER.info("All done!")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
