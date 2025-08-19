# ra_compiler/mysql.py

import os
import mysql.connector
from dotenv import load_dotenv
from .utils import print_error, clean_exit

conn = None
cursor = None

def setup_mysql(config_file=".env"):
    """Initialize the MySQL connection and cursor."""
    global conn, cursor

    try:
        print(f"Using configuration file: {config_file}")
        loaded = load_dotenv(dotenv_path=config_file)

        if not loaded:
            raise FileNotFoundError

        conn = connect()

        cursor = conn.cursor()

        print("MySQL Connection Successfully Complete")

    except FileNotFoundError as e:
        print_error(f"Error in loading config file {e.filename}", e)
        clean_exit(1)
    except mysql.connector.Error as e:
        print_error(f"Error creating cursor: {e}", e)
        clean_exit(1)

def connect():
    """Establish a connection to the MySQL database."""

    # make sure that the required fields are in the config file
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
    for var in required_vars:
        if os.getenv(var) is None:
            print_error(f"Missing required mysql config variable: {var}", "MySQLConfigError")
            clean_exit(1)

    try:
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
    
    except mysql.connector.Error as e:
        print_error(f"Database connection failed: {e}", e)
        clean_exit(1)

def close_mysql():
    """Closes the MySQL cursor and connection."""
    global conn, cursor

    try:
        if conn:
            conn.close()

        if cursor:
            cursor.close()
    finally:
        return

def run_query(sql):
    """Run a SQL query and return the results."""
    global conn, cursor

    # if there is no current connection, set one up
    if conn is None or cursor is None:
        setup_mysql()

    try:
        cursor.execute(sql)

        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    
        return columns, rows

    except mysql.connector.Error as e:
        handle_SQL_error(e, sql)
        return None

def handle_SQL_error(e, sql):
    if e.errno == 1146:
        print_error(f"SQL error: Table does not exist. {sql} ", e)
    else:
        print_error(f"SQL execution error: {e}", e)
        clean_exit(1)
