# ra_compiler/cli.py

import argparse
import pandas as pd
from .mysql import setup_mysql
from .parser import parse_query
from .translator import RATranslator
from .executor import execute, saved_results
from .utils import clean_exit, print_error, print_debug

def main():
    '''Main entry point for the RACompiler command line interface.'''

    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs="?", help="Path to the configuration file")
    args = parser.parse_args()
    
    # set up SQL connection based on the provided configuration file
    if args.config_file:
        print(f"Using configuration file: {args.config_file}")
        # TODO: set up SQL connection

    setup_mysql()
    run()

def run():
    '''Repeatedly handle user input, parses queries, and displays results.'''

    try:
        print("Welcome to RACompiler!")
        print("Type 'exit' to quit the application.")
        print("Type 'help' for a list of supported functions and syntax. [TODO]")

        query_counter = 0
        while True:
            # grab user input
            query = input("> ")

            # if the input is an exit command, cleanly exit the application
            exit_commands = ['exit', 'e', 'quit', 'q']
            if query.lower().strip(" /,.()") in exit_commands:
                clean_exit()

            result = handle_query(query, query_counter)
            if result is None:
                print("An error occurred while processing the query. Please try again.")
                continue
            
            # TODO: change to return the execution result in a separate window

            print("Execution Result:")
            print(result.name)
            print(result.df.to_string(index=False))
            #TODO: make a helper to print without index?
                        
            query_counter += 1


    except KeyboardInterrupt:
        clean_exit()
    except EOFError:
        clean_exit()
    except Exception as e:
        print_error(f"An Error Occurred: {e}", e)    
        clean_exit(1)

def handle_query(query, query_count=0):
    """Parse, translate, and execute a single query input."""

    print_debug(f"Handling query: {query}")

    parsed_query = parse_query(query)
    if parsed_query is None:
        return None
    # FOR TESTING: print the parsed query : Lark Tree
    pretty_parsed = parsed_query.pretty()
    print_debug(f"Parsed Query: {pretty_parsed}")
      
    # translate the parsed query into an intermediate representation
    translation = None
    try:
        translation = RATranslator(query_count).transform(parsed_query)
        # FOR TESTING:
        print_debug(f"Translation: {translation}")
    except Exception as e:
        print_error(f"An error occurred during translation: {e}", "TranslationError")
        return None

    # execute the translated query
    result = execute(translation)
    if result is None:
        return None

    # reset the index then save the result
    result.df = result.df.reset_index(drop=True)
    saved_results[result.name] = result
    return result
