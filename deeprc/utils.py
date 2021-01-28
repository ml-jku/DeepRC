# -*- coding: utf-8 -*-
"""
Utility functions and classes

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""
import os
import requests
import shutil
import tqdm


def user_confirmation(text: str = "Continue?", continue_if: str = 'y', abort_if: str = 'n'):
    """Wait for user confirmation"""
    while True:
        user_input = input(f"{text} ({continue_if}/{abort_if})")
        if user_input == continue_if:
            break
        elif user_input == abort_if:
            exit("Session terminated by user.")


def url_get(url: str, dst: str, verbose: bool = True):
    """Download file from `url` to file `dst`"""
    stream = requests.get(url, stream=True)
    try:
        stream_size = int(stream.headers['Content-Length'])
    except KeyError:
        raise FileNotFoundError(f"Sorry, the URL {url} could not be reached. "
                                f"Either your connection has a problem or the server is down."
                                f"Please check your connection, try again later, "
                                f"or notify me per email if the problem persists.")
    src = stream.raw
    windows = os.name == 'nt'
    copy_bufsize = 1024 * 1024 if windows else 64 * 1024
    update_progess_bar = tqdm.tqdm(total=stream_size, disable=not verbose,
                                   desc=f"Downloading {stream_size * 1e-9:0.3f}GB dataset")
    with open(dst, 'wb') as out_file:
        while True:
            buf = src.read(copy_bufsize)
            if not buf:
                break
            update_progess_bar.update(copy_bufsize)
            out_file.write(buf)
        shutil.copyfileobj(stream.raw, out_file)
    print()
    del stream
