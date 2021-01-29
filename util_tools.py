import time
import logging
import datetime

def print_fun_time(func):
    def run(*args, **kwargs):
        print("[Begin][{}][{}]".format(func.__name__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        if args:
            if kwargs:
                ret = func(*args, **kwargs)
            else:
                ret = func(*args)
        else:
            if kwargs:
                ret = func(**kwargs)
            else:
                ret = func()
        print("[End][{}][{}]".format(func.__name__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        return ret
    return run


def build_logger():
    # 初始化
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    # 设置输出格式
    # format_str = "[%(asctime)s] [%(filename)s %(funcName)s %(lineno)d] %(message)s "
    format_str = "[%(asctime)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    # 设置文件输出流
    log_file_name = datetime.datetime.today().strftime('%Y-%m-%d.log')
    file_handler = logging.FileHandler(os.path.join(data_path, log_file_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(format_str, datefmt))
    logger.addHandler(file_handler)
    # 设置屏幕输出流
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(format_str, datefmt))
    logger.addHandler(console_handler)

    return logger

logger = build_logger()

