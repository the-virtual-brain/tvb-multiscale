# coding=utf-8
# File writing/reading and manipulations

import os
from datetime import datetime
import glob
import shutil


def change_filename_or_overwrite(path, overwrite=True):
    if overwrite:
        if os.path.exists(path):
            os.remove(path)
        return path

    parent_folder = os.path.dirname(path)
    while os.path.exists(path):
        filename = \
            input("\n\nFile %s already exists. Enter a different name or press enter to overwrite file: " % path)
        if filename == "":
            overwrite = True
            break

        path = os.path.join(parent_folder, filename)

    if overwrite:
        os.remove(path)

    return path


def change_filename_or_overwrite_with_wildcard(path, overwrite=True):
    wild_path = path + "*"
    existing_files = glob.glob(path + "*")
    if len(existing_files) > 0:
        if overwrite:
            for file in existing_files:
                if os.path.exists(file):
                    os.remove(file)
            return path
        else:
            print("The following files already exist for base paths " + wild_path + " !: ")
            for file in existing_files:
                print(file)
            filename = input("\n\nEnter a different name or press enter to overwrite files: ")
            if filename == "":
                return change_filename_or_overwrite_with_wildcard(path, overwrite=True)
            else:
                parent_folder = os.path.dirname(path)
                path = os.path.join(parent_folder, filename)
                return change_filename_or_overwrite_with_wildcard(path, overwrite)
    else:
        return path
