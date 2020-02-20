import logging


def get_logger(name=None):
    logging.basicConfig(level=logging.INFO)
    if name is None:
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    return logger


if __name__ == '__main__':
    pass
