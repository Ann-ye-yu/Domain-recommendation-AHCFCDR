import logging
import time
import os
def getLogger(logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别，不设置默认为WARNING级
    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # # 创建控制台处理程序并设置日志级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    # # 创建文件处理程序并设置日志级别
    # local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # file_path = f"log/{local_time}-{logfile}"
    log_dir = logfile.split('0.')[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def getLogger_preprocess(logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别，不设置默认为WARNING级
    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # # 创建控制台处理程序并设置日志级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    # # 创建文件处理程序并设置日志级别
    # local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    # file_path = f"log/{local_time}-{logfile}"
    log_dir = logfile.split('/')[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger