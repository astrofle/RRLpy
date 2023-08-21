"""Console script for rrlpy."""

import argparse
import logging
import sys

LOGGER = logging.getLogger(__name__)


def foo():
    pass


def init_logging(level=logging.DEBUG):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    LOGGER.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    # See: https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=range(0, 4), default=1)
    return parser.parse_args()


def main():
    parse_args()
    foo()


if __name__ == "__main__":
    sys.exit(main())
