# utils/convert_logger.py
import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_file='logs/sketch_conversion_errors.log', level=logging.ERROR):
    """
    设置全局日志记录器
    :param log_file: 日志文件路径
    :param level: 日志级别（默认 ERROR）
    :return: 配置好的 logger 实例
    """
    # 创建日志目录（如果不存在）
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建 logger
    logger = logging.getLogger('SketchConversionLogger')
    logger.setLevel(level)

    # 防止重复添加 handler
    if not logger.handlers:
        # 创建 file handler 并设置级别
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(level)

        # 创建 formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 添加 formatter 到 handler
        file_handler.setFormatter(formatter)

        # 添加 handler 到 logger
        logger.addHandler(file_handler)

    return logger
