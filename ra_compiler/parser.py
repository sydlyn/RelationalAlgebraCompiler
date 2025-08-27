# ra_compiler/parser.py
'''Use the Lark parser and grammar to parse the query and handle errors.'''

import os
import sys
import re
from lark import Lark
import lark.exceptions
from .utils import clean_exit, print_error

# path to lark grammar file
GRAMMAR_FILE = 'grammar.lark'
PROJ_DIR = os.path.dirname(__file__)
GRAMMAR_PATH = os.path.join(PROJ_DIR, GRAMMAR_FILE)

# open the grammar file and set up the parser
try:
    with open(GRAMMAR_PATH, encoding="utf-8") as f:
        grammar_text = f.read()

    lark_parser = Lark(grammar_text, parser='earley', ambiguity='explicit')

except FileNotFoundError:
    print(f"Grammar file '{GRAMMAR_FILE}' not found at {PROJ_DIR}/. \
          Please check the path.", file=sys.stderr)
    clean_exit(1)
except lark.exceptions.LarkError as e:
    print(f"Error in grammar file '{GRAMMAR_FILE}': {e}", file=sys.stderr)
    clean_exit(1)

# parse a given query string into a Lark Tree object
def parse_query(query):
    """Return a lark Tree object representing the parsed query."""

    try:
        if not query:
            print("Empty query provided.")
            return None
        if not isinstance(query, str):
            print(f"Invalid query type: {type(query)}. Expected a string.")
            return None

        query = clean_query(query)
        parsed = lark_parser.parse(query)

        return parsed

    except lark.exceptions.UnexpectedInput as e:
        err_msg = handle_unexpected_input(query, e)
        print_error(f"Error Parsing Query: {err_msg}", "ParseError")

        print(f"{e.get_context(query)}")
        print("Please check the syntax of your query.")
        print("Type 'help' for a list of supported functions.")
    except Exception as e:
        print_error(f"An error occurred during parsing: {e}", e)
        clean_exit(1)

def clean_query(query):
    """Clean the input query to ensure it is ready to be parsed."""

    # remove leading/trailing whitespace and end line characters
    query = query.strip().strip(",;")

    # remove subscript underscores
    query = query.replace("_{", "{")

    # replace any backward slahsed keywords with forward slashes
    query = query.replace("\\", "/")

    return query

def handle_unexpected_input(query, e):
    """Return a helpful error message when handling unexpected input errors."""

    # print_debug("previous tokens:")
    # print_previous_tokens(query)

    if isinstance(e, lark.exceptions.UnexpectedEOF):
        return "Unexpected end of query. Check for missing operators or parentheses."

    if isinstance(e, lark.exceptions.UnexpectedCharacters):
        column = e.column
        bad_fragment = query[column-1:].split()[0] if query[column-1:].split() else query[column-1]

        if bad_fragment.startswith("/"):
            if not check_if_existing_operator(bad_fragment):
                return f"'{bad_fragment}' is not a valid operation."
            else:
                return f"Improper use of '{bad_fragment}' operator."

        if e.char == ")":
            return "Unmatched paretheses detected."

        if {"RENAME_ARROW"} == e.allowed:
            return f"Unexpected token at the end of the query: '{bad_fragment}'"

    expected = getattr(e, "expected", None)
    expected = getattr(e, "allowed", None) if not expected else None
    expected_msg = ""
    if expected:
        expected_msg= "\nExpected one of:\n"
        for s in expected:
            expected_msg += f"    * {s}\n"

    return f"Invalid query. {expected_msg}"


def print_previous_tokens(query):
    tokens = list(lark_parser.lex(query))

    for tok in tokens:
        print(tok)

def get_token_at_pos(query, pos):
    """Get the token at a specific position in the query."""

    tokens = list(lark_parser.lex(query))
    for token in tokens:
        if token.start_pos <= pos < token.end_pos:
            return token

    return None

def check_if_existing_operator(fragment):
    for term in lark_parser.terminals:
        if term.name.endswith("_PREFIX"):
            regex = re.compile(term.pattern.to_regexp(), re.IGNORECASE)
            if regex.fullmatch(fragment):
                return True
    return False
