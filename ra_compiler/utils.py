# ra_compiler/utils.py

import sys
import traceback
from . import mysql

DEBUG_COUNTER = 1

def print_error(err_msg="", etype="RACError"):
    '''Print the given error messgae.'''

    if isinstance(etype, Exception):
        etype = type(etype).__name__

    print(f"\n(RAC-ERR) [{etype}] {err_msg}", file=sys.stderr)
    # print(traceback.format_exc())

def print_warning(warn_msg="", e="RACWarning"):
    '''Print the given warning message.'''

    if isinstance(e, str):
        etype = e
    else:
        etype = type(e).__name__

    print(f"(RAC-WARN) [{etype}] {warn_msg}", file=sys.stderr)

def print_debug(debug_msg=""):
    '''Print a the given debug message.'''
    global DEBUG_COUNTER

    print(f"DEBUG {DEBUG_COUNTER}: {debug_msg}")

    DEBUG_COUNTER += 1

def clean_exit(exit_code=0):
    '''Cleanly exits the RACompiler application.'''

    print("\nExiting RACompiler...")

    try:
        mysql.close_mysql()
    finally:
        sys.exit(exit_code)
