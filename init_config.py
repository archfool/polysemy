import os
import logging
import datetime

if os.path.exists(u'c:\\'):
    root_path = u'E:\\'
elif os.path.exists(u'/media/archfool/data'):
    root_path = u'/media/archfool/data/'
elif os.path.exists(u'/mnt/hgfs/'):
    root_path = u'/mnt/hgfs/'

src_root_path = os.path.join(root_path, u'src')
data_root_path = os.path.join(root_path, u'data')

src_path = os.path.join(src_root_path, 'polysemy')
data_path = os.path.join(data_root_path, 'polysemy')

src_xml_path = os.path.join(src_root_path, 'XLM')
model_xml_path = os.path.join(data_root_path, 'XLM', 'mlm_17_1280')
task_corpus_path = os.path.join(data_root_path, 'SemEval2021', 'task 2', 'ver2')


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

def tmp():
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    filename = datetime.today().strftime('%Y-%m-%d-%H-%M-%S.log')
    logging.basicConfig(filename=filename,
                        level=logging.INFO,
                        format=format_str)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(format_str, "%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)


# print(tensor.device)
# print(torch.cuda.device_count())


if __name__ == "__main__":
    logger.info("一二三四五")
