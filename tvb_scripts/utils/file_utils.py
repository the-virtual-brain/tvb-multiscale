# -*- coding: utf-8 -*-
#
#
# File writing/reading and manipulations
#
#

import os
import glob
import shutil
import datetime


def ensure_unique_file(parent_folder, filename):
    final_path = os.path.join(parent_folder, filename)
    while os.path.exists(final_path):
        filename = eval(input("\n\nFile %s already exists. Enter a different name: " % final_path))
        final_path = os.path.join(parent_folder, filename)
    return final_path


def change_filename_or_overwrite(path, overwrite=True):
    if overwrite:
        if os.path.exists(path):
            os.remove(path)
        return path

    parent_folder = os.path.dirname(path)
    while os.path.exists(path):
        filename = eval(input("\n\nFile %s already exists. Enter a different name or press enter to overwrite: " % path))
        if filename == "":
            overwrite = True
            break

        path = os.path.join(parent_folder, filename)

    if overwrite:
        os.remove(path)

    return path


def wildcardit(name, front=True, back=True):
    out = str(name)
    if front:
        out = "*" + out
    if back:
        out = out + "*"
    return out


def ensure_folder(folderpath):
    if not os.path.isdir(folderpath):
        os.makedirs(folderpath)


def change_filename_or_overwrite_with_wildcard(path, overwrite=True):
    wild_path = path + "*"
    existing_files = glob.glob(path + "*")
    if len(existing_files) > 0:
        if overwrite:
            for file_ in existing_files:
                if os.path.exists(file_):
                    os.remove(file_)
            return path
        else:
            print("The following files already exist for base paths " + wild_path + " !: ")
            for file_ in existing_files:
                print(file_)
            filename = eval(input("\n\nEnter a different name or press enter to overwrite files: "))
            if filename == "":
                return change_filename_or_overwrite_with_wildcard(path, overwrite=True)
            else:
                parent_folder = os.path.dirname(path)
                path = os.path.join(parent_folder, filename)
                return change_filename_or_overwrite_with_wildcard(path, overwrite)
    else:
        return path


def move_overwrite_files_to_folder_with_wildcard(folder, path_wildcard):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for file_ in glob.glob(path_wildcard):
        if os.path.isfile(file_):
            filepath = os.path.join(folder, os.path.basename(file_))
            shutil.move(file_, filepath)


def write_metadata(meta_dict, h5_file, key_date, key_version, path="/"):
    root = h5_file[path].attrs
    root[key_date] = str(datetime.now())
    root[key_version] = 2
    for key, val in meta_dict.iteritems():
        root[key] = val
