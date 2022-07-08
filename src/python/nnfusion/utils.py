# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
import os
import logging
import subprocess
import hashlib

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


def get_sha256_of_file(path, max_len=None):
    hash_sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()[:max_len]


def get_sha256_of_str(string, max_len=None):
    return hashlib.sha256(string.encode("utf-8")).hexdigest()[:max_len]
