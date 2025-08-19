# ra_compiler/parser.py
'''Use the Lark parser and grammar to parse the query and handle errors.'''

import os
import sys
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

    #TODO: change to utilize LALR parser
    # lark_parser = Lark(grammar_text, parser='lalr')
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

    except lark.exceptions.UnexpectedToken as e:
        handle_unexpected_token(query, e)
    except lark.exceptions.UnexpectedInput as e:
        print_error(f"{e}", e)
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

# print helpful error messages based on the type of unexpected token encountered
def handle_unexpected_token(query, error):
    """Handle unexpected token errors."""

    try:
        # print("~~~~~~~~~~~~~~ UNEXPECTED TOKEN ~~~~~~~~~~~~~~~")

        # print(f"{error}")

        # print(f"error.expected: {error.expected}")
        # print(f"{error.accepts}")
        # print(error.pos_in_stream)
        # print(f"Token: {error.token}")
        # print("%r" % error.token)

        # print(error.token_history)
        # print(list(lark_parser.lex(query)))
        # for i, token in enumerate(lark_parser.lex(query)):
        #     print(token.type, token.value, token.start_pos)


        # print("~~~~~~~~~~~~~~ UNEXPECTED TOKEN ~~~~~~~~~~~~~~~")

        print_error("Error Parsing Query:", "ParseError")

        # get the tokens that are allowed at the error position
        allowed = error.accepts or error.expected

        # if the only allowed tokens are parentheses...
        if allowed in [{'LPAR'}, {'RPAR'}]:
            print("Missing parentheses.")

        # if an / is used incorrectly,
        # determine if it is an incorrect operator or a misused division operator
        elif 'MATH_OP' == error.token.type and error.token == "/":
            next_tok = get_token_at_pos(query, error.pos_in_stream + 1)
            if next_tok and next_tok.type == 'CNAME':
                print(f"Unknown Operator: {error.token}{next_tok.value}")
            else:
                print("Division operator '/' is not allowed in this context.")

        # if the end is reached unexpectedly...
        elif error.token.type == '$END':
            print("Unexpected end of query. Check for missing operators or parentheses.")

        # if the end is expected but more tokens are found...
        elif '$END' in allowed:
            print("Expected end of query, but found more tokens:", error.token)

        else:
            print("Unexpected token encountered.")

        # print helpful error information
        print(f"{error.get_context(query)}")
        print("Please check the syntax of your query.")
        print("Type 'help' for a list of supported functions.")

    except Exception as e:
        print(f"An error occurred while handling the unexpected token: {e}")
        clean_exit(1)

# parsing utility function to get the token at a specific position in the query
def get_token_at_pos(query, pos):
    """Get the token at a specific position in the query."""

    tokens = list(lark_parser.lex(query))
    for token in tokens:
        if token.start_pos <= pos < token.end_pos:
            return token

    return None
