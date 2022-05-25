import subprocess
import os
import glob
import json
import sys

def local2mountedRemote(local_dir, remote_dir, overwrite=False, options=''):
    """
    this function copies data from a local directory to a remote directory
    that is mounted onto the local file system. For copying the bash program
    RSYNC is being used.
    local_dir:      the local directory the data to be copied is stored in
    remote_dir:     the remote directory the data should be copied to (string)
    overwrite:      should already present files be overwritten? (True, False)
    options:        other options to be passed to rsync, arguments should be
                    passed as strings, e.g. options = '-u -z' for copying with
                    compression and only updating the remote file
    """
    if overwrite:
        return subprocess.check_output(f'rsync -rv {options} {local_dir}/ \
                                       {remote_dir}', shell=True)
    else:
        return subprocess.check_output(f'rsync -rv --ignore-existing \
                                       {options} {local_dir}/ {remote_dir}',
                                       shell=True)


def ssh_send(local_dir, remote_dir, user,
                 remote_address='storage.hpc.rz.uni-duesseldorf.de',
                 overwrite=False, options=''):
    """
    this function copies data from a local directory to a remote directory
    that is not yet mounted. To use this, you need to provide your user id and
    password.
    local_dir:      the local directory the data to be copied is stored in
    remote_dir:     the remote directory the data should be copied to (string)
    user:           the user that connects to the server
    remote_address: the address of the remote server (default is HILBERT)
    overwrite:      should already present files be overwritten? (True, False)
    options:        other options to be passed to rsync, arguments should be
                    passed as strings, e.g. options = '-u -z' for copying with
                    compression and only updating the remote file
    """

    if overwrite:
        return subprocess.check_output(f'rsync -av {options} {local_dir}/ \
                                       {user}@{remote_address}:{remote_dir}',
                                       shell=True)
    else:
        return subprocess.check_output(f'rsync -av --ignore-existing {options}\
                                       {local_dir}/ \
                                       {user}@{remote_address}:{remote_dir}',
                                       shell=True)
