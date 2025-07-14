# ra_compiler/executor.py

import pandas as pd
from .utils import clean_exit

def execute(ra, tables):
    '''Execute a given relational algebra query.'''

    try: 
        if not isinstance(ra, dict):
            raise TypeError("RA operation must be a dictionary representing the query.")
        if "op_type" not in ra:
            raise ValueError("RA operation must contain 'op_type' key.")
        
        # execute the operation based on its op_type
        match ra["operation"]:
            case "selection":
                return exec_selection(ra, tables)
            case "projection":
                return exec_projection(ra, tables)
            case "union":
                return exec_union(ra, tables)
            case "intersection":
                return exec_interestion(ra, tables)
            case "difference":
                return exec_difference(ra, tables)
            case "cross":
                return exec_cross(ra, tables)
            case _:
                raise ValueError(f"Unsupported operation type: {ra['operation']}")
    
    except Exception as e:
        print(f"An error occurred during execution. {type(e).__name__}. {type(e).__name__}: {e}")
        clean_exit(1)

def get_table(table_name, tables):
    # if the table is a string, grab the table from SQL connection
    if isinstance(table_name, str):
        # TODO: get table from SQL connection 
        # + open table as a pandas DataFrame
        # + error check if table exists
        # for now, using dummy data passed in as tables
        
        return (tables[table_name]).copy()
    else: 
        return (execute(table_name, tables)).copy()
    

## ~~~~~~~~ UNARY OPERATIONS ~~~~~~~~ ##

def exec_projection(ra, tables):
    """Execute a projection operation on the given DataFrame."""

    try:
        df = get_table(ra["table"], tables)
        attributes = []
        keep_dups = ra.get("keep_dups", False)

        for i, attr in enumerate(ra["attributes"]):
            if isinstance(attr, dict):
                # Evaluate the math condition and add the result as a new column
                df[attr["alias"]] = evaluate_math_condition(df, attr)
                attributes.append(attr["alias"])
            else:
                attributes.append(attr)
                        
        # select the specified attributes
        result_df = df[attributes]
        
        # handle duplicates if necessary
        if not keep_dups:
            result_df = result_df.drop_duplicates()
        
        return result_df
    
    except Exception as e:
        print(f"An error occurred during projection execution. {type(e).__name__}: {e}")
        clean_exit(1)

def exec_selection(ra, tables):
    """Execute the given selection operation."""
    
    try:
        df = get_table(ra["table"], tables)
        cond = ra["condition"]

        mask = evaluate_comparison_condition(df, cond)

        return df[mask]
    
    except Exception as e:
        print(f"An error occurred during selection execution. {type(e).__name__}: {e}")
        clean_exit(1)


## ~~~~~~~~ SET OPERATIONS ~~~~~~~~ ##

def exec_union(ra, tables):
    """Execute the given union operation."""

    try:
        df1 = get_table(ra["table1"], tables)
        df2 = get_table(ra["table2"], tables)
        keep_dups = ra.get("keep_dups", False)

        # combine the two sets
        result_df = pd.concat([df1, df2], ignore_index=True)
    
        # handle duplicates if necessary
        if not keep_dups:
            result_df = result_df.drop_duplicates()

        return result_df
        
    except Exception as e:
        print(f"An error occurred during union execution. {type(e).__name__}: {e}")
        clean_exit(1)

def exec_interestion(ra, tables):
    """Execute the given intersection operation."""

    try:
        df1 = get_table(ra["table1"], tables)
        df2 = get_table(ra["table2"], tables)
        keep_dups = ra.get("keep_dups", False)

        # remove duplicates if not a bag intersection
        if not keep_dups:
            df1_dr = df1.drop_duplicates()
            df2_dr = df2.drop_duplicates()

        # if a bag intersection, add row counts to keep desired duplication
        else:
            def add_row_counts(df):
                return df.assign(
                    _rownum=df.groupby(df.columns.tolist()).cumcount()
                )
           
            df1_dr = add_row_counts(df1)
            df2_dr = add_row_counts(df2)
            
        # merge the data frames and keep only the relevant columns
        inter = pd.merge(df1_dr, df2_dr)
        result_df = inter[df1.columns]

        return result_df
        
    except Exception as e:
        print(f"An error occurred during intersection execution. {type(e).__name__}: {e}")
        clean_exit(1)  

def exec_difference(ra, tables):
    """Execute the given difference operation."""

    try:
        df1 = get_table(ra["table1"], tables)
        df2 = get_table(ra["table2"], tables)
        keep_dups = ra.get("keep_dups", False)

        # remove duplicates if not a bag difference
        if not keep_dups:
            df1_dr = df1.drop_duplicates()
            df2_dr = df2.drop_duplicates()

        # if a bag intersection, add row counts to keep desired duplication
        else:
            def add_row_counts(df):
                return df.assign(
                    _rownum=df.groupby(df.columns.tolist()).cumcount()
                )
           
            df1_dr = add_row_counts(df1)
            df2_dr = add_row_counts(df2)

        # merge the data frames and keep only the relevant columns/rows
        merged = pd.merge(df1_dr, df2_dr, how="left", indicator=True)
        diff = merged[merged['_merge'] == 'left_only']
        result_df = diff[df1.columns]

        return result_df
        
    except Exception as e:
        print(f"An error occurred during difference execution. {type(e).__name__}: {e}")
        clean_exit(1)  


## ~~~~~~~~ JOIN OPERATIONS ~~~~~~~~ ##

def exec_cross(ra, tables):
    """Execute the given cross operation."""

    try:
        df1 = get_table(ra["table1"], tables)
        df2 = get_table(ra["table2"], tables)

        result_df = pd.merge(df1, df2, how="cross")

        return result_df
    
    except Exception as e:
        print(f"An error occurred during cross execution. {type(e).__name__}: {e}")
        clean_exit(1)

def exec_join(ra, tables):
    """Execute the given join operation."""

    try:
        df1 = get_table(ra["table1"], tables)
        df2 = get_table(ra["table2"], tables)

        


    except Exception as e:
        print(f"An error occurred during cross execution. {type(e).__name__}: {e}")
        clean_exit(1)

## ~~~~~~~~ TABLES, ATTRIBUTES, & OTHER ~~~~~~~~ ##


def evaluate_math_condition(df, attr):
    """Evaluate a math condition and add the result as a new column."""
    left = attr['cond']["left"]
    op = attr['cond']["op"]
    right = attr['cond']["right"]

    if isinstance(right, int):
        right = int(right)
    elif right.isidentifier():
        right = df[right]

    if op == "+":
        return df[left] + right
    elif op == "-":
        return df[left] - right
    elif op == "*":
        return df[left] * right
    elif op == "/":
        return df[left] / right
    else:
        raise ValueError(f"Unsupported math operation: {op}")

def evaluate_comparison_condition(df, cond):
    """Evaluate a comparison condition and return a boolean mask."""
    left = cond["left"]
    op = cond["op"]
    right = cond["right"]

    if op == ">":
        return df[left] > right
    elif op == "<":
        return df[left] < right
    elif op == "=" or op == "==":
        return df[left] == right
    elif op == ">=":
        return df[left] >= right
    elif op == "<=":
        return df[left] <= right
    elif op == "!=":
        return df[left] != right
    elif op == "and":
        left_mask = evaluate_comparison_condition(df, cond["left"])
        right_mask = evaluate_comparison_condition(df, cond["right"])
        return left_mask & right_mask
    elif op == "or":
        left_mask = evaluate_comparison_condition(df, cond["left"])
        right_mask = evaluate_comparison_condition(df, cond["right"])
        return left_mask | right_mask
    else:
        raise ValueError(f"Unsupported comparison operator: {op}")
