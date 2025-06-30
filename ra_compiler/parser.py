# ra_compiler/parser.py

import os, sys
from lark import Lark
import lark.exceptions

# path to lark grammar file
GRAMMAR_FILE = 'grammar.lark'
PROJ_DIR = os.path.dirname(__file__)
GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), GRAMMAR_FILE)

# open the grammar file and set up the parser
try: 
    with open(GRAMMAR_PATH) as f:
        grammar_text = f.read()
        
    lark_parser = Lark(grammar_text, parser='lalr')

except FileNotFoundError:
    print(f"Grammar file '{GRAMMAR_FILE}' not found at {PROJ_DIR}/. Please check the path.", file=sys.stderr)
    exit(1)
except lark.exceptions.LarkError as e:
    print(f"Error in grammar file '{GRAMMAR_FILE}': {e}", file=sys.stderr)
    exit(1)

def parse_query(query):
    """Return a lark Tree object representing the parsed query."""

    if not query:
        print("Empty query provided.")
        return None
    if not isinstance(query, str):
        print(f"Invalid query type: {type(query)}. Expected a string.")
        return None

    try:
        parsed = lark_parser.parse(query)

        return parsed

    except lark.exceptions.UnexpectedInput as e:
        print(f"Unexpected input in query: {query}\nError: {e}")
    except lark.exceptions.UnexpectedToken as e:
        print(f"Unexpected token in query: {query}\nError: {e}")
    except Exception as e:
        print(f"Failed to parse query: {query}\nError: {e}")
