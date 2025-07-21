# ra_compiler/cli.py

import argparse
import pandas as pd
from .mysql import setup_mysql
from .parser import parse_query
from .translator import RATranslator
from .executor import execute
from .utils import clean_exit, print_error

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

            # if given a bad input for parsing, continue to the next iteration
            parsed_query = parse_query(query)
            if parsed_query is None:
                continue
            # FOR TESTING: print the parsed query : Lark Tree
            print("Parsed Query: ", parsed_query.pretty())

            # translate the parsed query into an intermediate representation
            translation = None
            try:
                translation = RATranslator(query_counter).transform(parsed_query)
            except Exception as e:
                print_error(f"An error occurred during translation: {e}", "TranslationError")
                continue
            # FOR TESTING: print the translation
            print("translator: ", translation, "\n")

            # execute the translated query
            result = execute(translation)
            if result is None:
                continue

            table_name, table_result = result
            if table_result is None:
                continue
            
            # TODO: change to return the execution result in a separate window
            print("Execution Result:")
            print(table_name)
            print(table_result)
                        
            query_counter += 1


    except KeyboardInterrupt:
        clean_exit()
    except EOFError:
        clean_exit()
    except Exception as e:
        print_error(f"An Error Occurred: {e}", e)    
        clean_exit(1)
