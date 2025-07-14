# ra_compiler/cli.py

import argparse
import pandas as pd
from .parser import parse_query
from .translator import RATranslator
from .executor import execute
from .utils import clean_exit

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

    run()


def run():
    '''Repeatedly handle user input, parses queries, and displays results.'''

    try:
        print("Welcome to RACompiler!")
        print("Type 'exit' to quit the application.")
        print("Type 'help' for a list of supported functions and syntax. [TODO]")

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
            translation = RATranslator().transform(parsed_query)
            # FOR TESTING: print the translation
            print("translator: ", translation)

            # TODO: set up SQL connection to grab actual tables, for now dummy data
            tables = {
                "Students": pd.read_csv("test.csv"),
                "T1": pd.read_csv("test.csv"),
                "T2": pd.read_csv("test2.csv"),
                "T3": pd.read_csv("test.csv").T,
                "A": pd.read_csv("test.csv"),
                "B": pd.read_csv("test.csv")['age'].add(2),
                "C1": pd.read_csv("test.csv"),
                "C2": pd.read_csv("test.csv").T,
                "C3": pd.read_csv("test.csv")['age'].add(3),
                "R": pd.read_csv("test.csv"),
            }

            # execute the translated query
            result = execute(translation, tables)
            # TODO: change to return the execution result in a separate window
            print("Execution Result:")
            print(result)

    except KeyboardInterrupt:
        clean_exit()
    except EOFError:
        clean_exit()
    except Exception as e:
        print(f"An Error Occurred: {e}")    
        clean_exit(1)
