# ra_compiler/exceptions.py
'''Custom exceptions for RACompiler.'''

class RacException(Exception):
    """Base exception for RACompiler errors."""
    def __init__(self, message="An error occurred while trying to process the query."):
        super().__init__(message)

class NestedQueryError(RacException):
    def __init__(self):
        super().__init__()

class TableNotFoundError(RacException):
    def __init__(self, table_name=""):
        super().__init__(f"Table '{table_name}' not found.")
        self.table_name = table_name

class TableAlreadyExists(RacException):
    def __init__(self, table_name=""):
        super().__init__(f"Table '{table_name}' already exists.")
        self.table_name = table_name

class InvalidColumnName(RacException):
    def __init__(self, col_name):
        super().__init__(f"Column name '{col_name}' not found in table.")
        self.col_name = col_name
