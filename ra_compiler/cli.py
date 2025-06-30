# ra_compiler/cli.py

import argparse
from .parser import parse_query

def main():
    '''Main entry point for the RACompiler command line interface.'''

    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs="?", help="Path to the configuration file")
    args = parser.parse_args()

    # parse the test file 
        # with open("tests/test_grammar.txt", 'r') as f:
        #     lines = f.read()

        # for line in lines.strip().splitlines():
        #     parsed_query = parse_query(line)
        #     if parsed_query is not None:
        #         print("Parsed Query: ", parsed_query.pretty())
        #     else:
        #         print("Failed to parse the query.")

    run()


def run():
    try:
        print("Welcome to RACompiler!")
        print("Type 'exit' to quit the application.")
        print("Type 'help' to display a list of supported functions and syntax. [TODO]")

        while True:
            query = input("> ")

            exit_commands = ['exit', 'e', 'quit', 'q']
            if query.lower().strip(" /,.()") in exit_commands:
                clean_exit()

            parsed_query = parse_query(query)

            if parsed_query is None:
                print("Failed to parse the query.")
                continue
                
            print("Parsed Query: ", parsed_query.pretty())

    except KeyboardInterrupt:
        clean_exit()
    except EOFError:
        clean_exit()
    except Exception as e:
        print(f"An Error Occurred: {e}")    
        clean_exit(1)

def clean_exit(exit_code=0):
    '''Cleanly exits the RACompiler application.'''
    print("\nExiting RACompiler...")
    exit(exit_code)
