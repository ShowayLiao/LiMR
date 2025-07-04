# modified from https://github.com/zhangzilongc/MMR
# copyright (c) 2023 Z.Zhang et al.
# Licensed under the Apache License, Version 2.0 (the "LICENSE-APACHE-2.0-ZHANG2023");

# copyright (c) 2025 S.Liao et al.


import logging
import time
import sys
import os


def setup_logging(cfg):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """

    output_file_name = cfg.OUTPUT_DIR

    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # 移除所有现有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    # 设置控制台输出handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)

    # 设置文件输出handler
    filename = os.path.join(output_file_name, 'run.log')
    file_info_handler = logging.FileHandler(filename)
    file_info_handler.setLevel(logging.INFO)
    file_info_handler.setFormatter(plain_formatter)
    logger.addHandler(file_info_handler)

    # 更新配置中的输出目录
    cfg.OUTPUT_DIR = output_file_name


def set_output_dir(cfg):
    """
    Set the output directory.
    """
    # Set up time format
    timeArray = time.localtime()
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M", timeArray)
    otherStyleTime = otherStyleTime.replace(" ", "_").replace("--", "_").replace(":", "_")


    # Set up the output directory by run mode
    if cfg.TRAIN.enable and cfg.TRAIN.resume is not True:
        output_file_name = os.path.join(cfg.OUTPUT_ROOT_DIR,
                                        cfg.DATASET.name + f"{cfg.DATASET.imagesize}",
                                        cfg.TRAIN.method,
                                        cfg.TRAIN.change,
                                        str(cfg.RNG_SEED) + "_" + otherStyleTime
                                        )
    elif cfg.TRAIN.enable and cfg.TRAIN.resume:
        # test和resume指定读取的路径都是一样的文件，选择文件上的路径
        output_file_name = os.path.split(cfg.TRAIN.resume_model_path)[0]
    else:
        output_path = os.path.split(cfg.TEST.model_path)[0]
        output_file_name = os.path.join(output_path, "test_output")
        print(f"Test output directory: {output_file_name}")

    if not os.path.exists(output_file_name):
        try:
            os.makedirs(output_file_name)
        except OSError:
            raise OSError(f"Failed to create output directory: {output_file_name}")

    cfg.OUTPUT_DIR = output_file_name

    return cfg
def get_max_epoch_pth_file(directory):
    # Initialize variables to keep track of the file with the maximum epoch
    max_epoch = -1
    max_epoch_file = None

    # Loop through all files in the given directory
    for file in os.listdir(directory):
        # Check if the file is a .pth file
        if file.endswith('.pth'):
            # Extract the epoch number from the file name
            epoch_str = file.split('_epoch_')[-1].split('.pth')[0]
            try:
                epoch = int(epoch_str)
                # Update the maximum epoch and file name if necessary
                if epoch > max_epoch:
                    max_epoch = epoch
                    max_epoch_file = file
            except ValueError:
                # Skip files that do not have a valid epoch number
                continue

    return os.path.join(directory,max_epoch_file)

