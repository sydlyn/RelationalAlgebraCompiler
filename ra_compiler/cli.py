# ra_compiler/cli.py

import argparse
import sys
import pathlib
from rich.console import Console
from rich.table import Table
from .mysql import setup_mysql
from .parser import parse_query
from .translator import RATranslator
from .executor import execute, saved_results
from .utils import clean_exit, print_error, print_debug

app = None
open_windows = []
qt_thread = None

def main():
    '''Main entry point for the RACompiler command line interface.'''

    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", nargs="?", help="Path to the SQL backend configuration file")
    args = parser.parse_args()

    # grab the config file if given one, else default to ".env"
    config = args.config_file if args.config_file else ".env"

    setup_mysql(config)

    # app = QApplication(sys.argv)
    run()
    sys.exit(app.exec_())

def run():
    '''Repeatedly handle user input, parses queries, and displays results.'''

    try:
        print("\nWelcome to RACompiler!")
        print("Type 'exit' to quit the application.")
        print("Type 'help' for a list of supported functions and syntax.")

        query_counter = 0
        while True:
            # grab user input
            query = input("> ")

            # if the input is an exit command, cleanly exit the application
            exit_commands = ['exit', 'e', 'quit', 'q']
            if query.lower().strip(" /,.()") in exit_commands:
                clean_exit()

            help_commands = ['help', 'h', '-h', '-help']
            if query.lower().strip(" /,.()") in help_commands:
                file_path = 'docs/quick_reference.txt' 
                try:
                    with open(file_path, 'r') as file:
                        content = file.read()
                        print(content)
                except FileNotFoundError:
                    print(f"Error: File '{file_path}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                continue

            result = handle_query(query, query_counter)
            if result is None:
                print("An error occurred while processing the query. Please try again.")
                continue
            
            
            # print the results without the df index
            print("Execution Result:")
            print(result.name)
            print(result.df.to_string(index=False))

            path = pathlib.Path(f"out/{result.name}.csv").absolute()
            result.df.to_csv(path, index=False)

            # output the results in a separate window?
            show_dataframe(result.name, result.df)
                        
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

# pretty print the table 
def show_dataframe(df_name, df):
    console = Console()
    table = Table(title=df_name)
    for col in df.columns:
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(*map(str, row))
    
    console.print(table)
