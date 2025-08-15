# ra_compiler/utils.py

import sys, traceback
from . import mysql

debug_counter = 1

def print_error(err_msg="", e="RACError"):
    '''Print the given error messgae.'''
    if isinstance(e, str):
        etype = e
    else:
        etype = type(e).__name__

    print(f"\n(RAC-ERR) [{etype}] {err_msg}", file=sys.stderr)
    print(traceback.format_exc())
    return

def print_warning(warn_msg="", e="RACWarning"):
    '''Print the given warning message.'''
    if isinstance(e, str):
        etype = e
    else:
        etype = type(e).__name__

    print(f"(RAC-WARN) [{etype}] {warn_msg}", file=sys.stderr)
    return

def print_debug(debug_msg=""):
    '''Print a the given debug message.'''
    global debug_counter
    print(f"DEBUG {debug_counter}: {debug_msg}")
    debug_counter += 1
    return

def clean_exit(exit_code=0):
    '''Cleanly exits the RACompiler application.'''

    print("\nExiting RACompiler...")

    try:
        mysql.close_mysql()
    finally:
        exit(exit_code)
