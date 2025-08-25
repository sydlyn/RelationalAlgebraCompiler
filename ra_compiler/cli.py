# ra_compiler/cli.py
"""The command line interface handler. Handles program set up and user input."""

import os
import argparse
import pathlib
import atexit
import pandas as pd
from rich.console import Console
from rich.table import Table
from .mysql import setup_mysql
from .parser import parse_query
from .translator import RATranslator
from .executor import execute, saved_results
from .utils import clean_exit, print_error, print_debug

# import windows equivalent of readline
try:
    import readline  # Unix / macOS
except ImportError:
    # Windows fallback
    import pyreadline3 as readline


def main():
    """Main entry point for the RACompiler command line interface."""

    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs="?", default=".env",
                        help="path to the SQL backend configuration file")
    parser.add_argument("-out", action="store_true",
                        help="save output tables as CSVs to the out/ folder")

    # parse the command line arguments
    args = parser.parse_args()
    rac_setup(args)

def rac_setup(args):
    """Set up the database connection and user interface at start up."""

    # set up the sql database connection
    setup_mysql(args.config_file)

    # display the start up messages
    print("\nWelcome to RACompiler!")
    print("Type 'exit' to quit the application.")
    print("Type 'help' for a list of supported functions and syntax.")

    # set up for the cli history to view previous queries
    history_file = ".rac_cache/ra_history"

    # create the path to the history file if it doesn't exist yet
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    if os.path.exists(history_file):
        readline.read_history_file(history_file)
    readline.set_history_length(50)
    atexit.register(readline.write_history_file, history_file)

    run(save_to_out=args.out)

def run(save_to_out=False, query_counter=0):
    """Repeatedly handle user input, parses queries, and displays results."""

    try:
        while True:
            # grab user input
            query = input("> ")

            # check if the command was a help/exit request
            if check_if_help_command(query):
                continue

            # if something goes wrong handling the query, skip to the next one
            result = handle_query(query, query_counter)
            if result is None:
                continue

            # if the result is a table list, print our the tables
            if isinstance(result, list):
                print("List of tables:")
                for table in result:
                    print(f" - {table}")
                continue

            # otherwise, cleanly output the datafram results
            print("Execution Result:")
            show_dataframe(result.name, result.df)

            # if specified, save the result to a csv file in the out/ folder
            if save_to_out:
                path = pathlib.Path(f"out/{result.name}.csv").absolute()
                result.df.to_csv(path, index=False)

            query_counter += 1

    except KeyboardInterrupt:
        clean_exit()
    except EOFError:
        clean_exit()
    except Exception as e:
        print_error(f"An Error Occurred: {e}", e)
        run(query_counter+1)

def check_if_help_command(query):
    """Handle any 'exit' or 'help' commands. Return true if 'help'."""

    # if the input is an exit command, cleanly exit the application
    exit_commands = ['exit', 'e', 'quit', 'q']
    if query.lower().strip(" /,.()") in exit_commands:
        clean_exit()

    # if the input is a help command, print out the quick reference doc
    help_commands = ['help', 'h', '-h', '-help']
    if query.lower().strip(" /,.()") in help_commands:
        file_path = 'docs/quick_reference.txt'
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
            print(content)
        return True

def handle_query(query, query_count=0):
    """Parse, translate, and execute a single query input."""

    # print_debug(f"query: {query}")

    parsed_query = parse_query(query)
    if parsed_query is None:
        return None

    # FOR TESTING: print the parsed query : Lark Tree
    # pretty_parsed = parsed_query.pretty()
    # print_debug(f"Parsed Query: {pretty_parsed}")

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

    # if the result is a list, return the list
    if isinstance(result, list):
        return result

    # otherwise, reset the index then save the dataframe result
    result.df = result.df.reset_index(drop=True)
    saved_results[result.name] = result
    return result

def show_dataframe(df_name, df):
    """Nicely print out a pandas DataFrame with a corresponding name."""

    # convert the columns to nullable types for consistency
    df.convert_dtypes()

    # use the rich Console to display the table
    console = Console()
    table = Table(title=df_name)

    for col in df.columns:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(*[
            "<NA>" if pd.isna(v) else str(v)
            for v in row
        ])
    console.print(table)
