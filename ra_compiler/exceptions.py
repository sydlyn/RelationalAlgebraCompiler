# ra_compiler/exceptions.py

class TableNotFoundError(Exception):
    def __init__(self, table_name):
        super().__init__(f"Table '{table_name}' not found.")
        self.table_name = table_name
    
class InvalidColumnName(Exception):
    def __init__(self, col_name):
        super().__init__(f"Column name '{col_name}' not found in table.")
        self.col_name = col_name
