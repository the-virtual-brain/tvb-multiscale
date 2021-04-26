# -*- coding: utf-8 -*-

def truncate_ascii_file_after_header(filepath, header_chars="#"):
    n_header_chars = len(header_chars)
    with open("/Users/dionperd/Desktop/test.txt", "r+") as file:
        line = file.readline()
        while len(line) >= n_header_chars and line[:n_header_chars] == header_chars:
            line = file.readline()
        file.seek(file.tell() - len(line), os.SEEK_SET)
        file.truncate()
        file.close()
