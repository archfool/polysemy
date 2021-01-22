import time
import torch

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


def check_gpu(x):
    if x is None:
        return None
    elif torch.cuda.is_available():
        return x.cuda()
    else:
        return x
