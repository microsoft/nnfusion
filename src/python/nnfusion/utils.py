# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
import os
import logging
import subprocess

logger = logging.getLogger(__name__)


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def execute(command, redirect_stderr=True, shell=True, **kwargs):
    logger.debug(command)
    stderr = subprocess.STDOUT if redirect_stderr else None
    try:
        output = subprocess.check_output(command,
                                         stderr=stderr,
                                         shell=shell,
                                         encoding="utf8",
                                         **kwargs)
    except subprocess.CalledProcessError as e:
        logger.error(e.output)
        raise e
    return output