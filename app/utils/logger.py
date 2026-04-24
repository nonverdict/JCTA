import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
    )
    return logging.getLogger("PoultryScope")


logger = setup_logger()
