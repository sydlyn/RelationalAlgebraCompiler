# ra_compiler/executor.py

import pandas as pd
from .mysql import run_query
from .utils import clean_exit, print_error, print_debug
from .exceptions import *

saved_results = {}

class NamedDataFrame():
    def __init__(self, name, df, origin_name=None):
        self.name = name
        self.df = df
        self.origin_name = origin_name

    def __repr__(self):
        return f"NamedDataFrame(name={self.name}, origin_name={self.origin_name}, shape={self.df.shape})"
    
    def __str__(self):
        return f"NamedDataFrame: {self.name} (origin: {self.origin_name}, shape: {self.df.shape})"

def execute(expr):
    '''Execute a given relational algebra query and return the table name and table as a DataFrame.'''

    operation = None

    try: 
        if not isinstance(expr, dict):
            raise TypeError("RA expression must be a dictionary representing the query.")
        if "op_type" not in expr or "operation" not in expr:
            raise ValueError("RA expression must contain 'op_type' and 'operation' key.")

        operation = expr["operation"]
        df, df2 = get_tables(expr)

        # execute the operation
        match operation:
            case "projection":
                return exec_projection(expr, df)
            case "selection":
                return exec_selection(expr, df)
            case "group":
                return exec_group(expr, df)
            case "rename": 
                return exec_rename(expr, df)
            case "remove_duplicates":
                return exec_remove_duplicates(expr, df)
            case "sort":
                return exec_sort(expr, df)
            case "union":
                return exec_union(expr, df, df2)
            case "intersection":
                return exec_interestion(expr, df, df2)
            case "difference":
                return exec_difference(expr, df, df2)
            case "cross":
                return exec_cross(expr, df, df2)
            case "join":
                return exec_join(expr, df, df2)
            case "divide":
                return exec_divide(expr, df, df2)
            case _:
                raise ValueError(f"Unsupported operation: {expr['operation']}")
            
    except TableNotFoundError as e:
        print_error(f"Table '{e.table_name}' does not exist in the database.", e)
        return None
    except Exception as e:
        if not operation:
            operation = "query"

        print_error(f"An error occurred during {operation} execution: {e}", e)
        return None

def get_tables(expr):
    """Ensure the desired table(s) exist depending on the op_type."""
    
    match expr["op_type"]:
        case "unary":
            df = load_table(expr["table"])
            return df, None
        case "set":
            df = load_table(expr["table1"])
            df2 = load_table(expr["table2"])

            if not df.df.columns.equals(df2.df.columns):
                raise ValueError("Set operations require both tables to have the same columns in the same order.")

            return df, df2
        case "join":
            df = load_table(expr["table1"])
            df2 = load_table(expr["table2"])

            return df, df2
        case _:
            raise ValueError(f"Unsupported operation type: {expr['op_type']}")

def load_table(table_name):
    """Return a pandas DataFrame of the desired table from the SQL connection."""

    # if the table is a string...
    if isinstance(table_name, str):   

        # check if a saved result  
        if table_name in saved_results:
            return saved_results[table_name]
         
        # or check if it is a table from SQL connection
        check_query = f"SHOW TABLES LIKE '{table_name}'"
        result = run_query(check_query)
        
        # if the table does not exist, error
        if len(result[1]) == 0:
            raise TableNotFoundError(table_name)

        # if the table exists, load and save it
        query = f"SELECT * FROM `{table_name}`"
        cols, rows = run_query(query)
        df = pd.DataFrame(rows, columns=cols)

        ndf = NamedDataFrame(table_name, df.copy(), table_name)
        saved_results[table_name] = ndf

        return ndf
    
    # otherwise, execute the nested operation
    else: 
        ndf = (execute(table_name))
        return ndf
    

## ~~~~~~~~ UNARY OPERATIONS ~~~~~~~~ ##

def exec_projection(expr, ndf):
    """Execute a projection operation on the given DataFrame."""

    df = ndf.df
    keep_dups = expr.get("keep_dups", False)
    attributes = process_attributes(df, expr["attributes"])

    result_df = df[attributes]
    
    # handle duplicates if necessary
    if not keep_dups:
        result_df = result_df.drop_duplicates()
    
    return NamedDataFrame(expr['table_alias'], result_df, ndf.origin_name)


def exec_selection(expr, ndf):
    """Execute the given selection operation on the given DataFrame."""
    
    df = ndf.df
    cond = expr["condition"]
    mask = evaluate_comparison_cond(df, cond)

    return NamedDataFrame(expr['table_alias'], df[mask], ndf.origin_name)


def exec_group(expr, ndf):
    """Execute the given group operation on the given DataFrame."""

    df = ndf.df
    group_attrs = process_attributes(df, expr["attributes"])
    aggr_funcs = parse_aggr_conds(expr["aggr_cond"])

    group_df = df.groupby(group_attrs, sort=False)

    # Construct aggregation mapping for pandas
    agg_dict = {}
    for alias, (col, func) in aggr_funcs.items():
        if col == "*":
            # Special case for count(*): use .size()
            agg_dict[alias] = ("", "size")
        else:
            agg_dict[alias] = pd.NamedAgg(column=col, aggfunc=func)

    result_df = group_df.agg(**agg_dict).reset_index()
    return NamedDataFrame(expr['table_alias'], result_df)


def exec_rename(expr, ndf):
    """Rename the table to the given alias."""
    # TODO: check if the table alias already exists

    return NamedDataFrame(expr["table_alias"], ndf.df, expr["table_alias"])

def exec_remove_duplicates(expr, ndf):
    """Remove duplicates from the given DataFrame."""

    return NamedDataFrame(expr["table_alias"], ndf.df.drop_duplicates(), ndf.origin_name)

def exec_sort(expr, ndf):
    """Sort the DataFrame based on the given attributes."""
    
    df = ndf.df
    sort_attrs = expr["sort_attributes"]

    for attr in sort_attrs:
        # ensure the attribute is a valid column in the DataFrame
        col = attr[0]
        if col not in df.columns:
            raise InvalidColumnName(col)
        
        result_df = result_df.sort_values(by=col, ascending=attr[1])

    return NamedDataFrame(expr['table_alias'], result_df, ndf.origin_name)


## ~~~~~~~~ SET OPERATIONS ~~~~~~~~ ##

def exec_union(expr, df1, df2):
    """Execute the given union operation on the given DataFrames."""

    keep_dups = expr.get("keep_dups", False)

    # combine the two sets
    result_df = pd.concat([df1.df, df2.df], ignore_index=True)

    # handle duplicates if necessary
    if not keep_dups:
        result_df = result_df.drop_duplicates()

    return NamedDataFrame(expr['table_alias'], result_df)
        

def exec_interestion(expr, df1, df2):
    """Execute the given intersection operation on the given DataFrames."""

    keep_dups = expr.get("keep_dups", False)
    df1_dr, df2_dr = prepare_for_set_op(df1.df, df2.df, keep_dups)
        
    # merge the data frames and keep only the relevant columns
    inter = pd.merge(df1_dr, df2_dr)
    result_df = inter[df1.df.columns]

    return NamedDataFrame(expr['table_alias'], result_df)


def exec_difference(expr, df1, df2):
    """Execute the given difference operation on the given DataFrames."""

    keep_dups = expr.get("keep_dups", False)
    df1_dr, df2_dr = prepare_for_set_op(df1.df, df2.df, keep_dups)

    # merge the data frames and keep only the relevant columns/rows
    merged = pd.merge(df1_dr, df2_dr, how="left", indicator=True)
    diff = merged[merged['_merge'] == 'left_only']
    result_df = diff[df1.df.columns]

    return NamedDataFrame(expr["table_alias"], result_df)


def prepare_for_set_op(df1, df2, keep_dups=False):
    """Prepares two DataFrames for set operations by either dropping duplicates or adding row counts (for bag semantics)."""

    # remove duplicates if not a bag
    if not keep_dups:
        df1_clean = df1.drop_duplicates()
        df2_clean = df2.drop_duplicates()

    # if a bag, add row counts to keep desired duplication
    else:
        def add_row_counts(df):
            '''Add row counts to the rows.'''
            return df.assign(
                _rownum=df.groupby(df.columns.tolist()).cumcount()
            )
        df1_clean = add_row_counts(df1)
        df2_clean = add_row_counts(df2)
    
    return df1_clean, df2_clean

## ~~~~~~~~ JOIN OPERATIONS ~~~~~~~~ ##

def exec_cross(expr, df1, df2):
    """Execute the given cross operation on the given DataFrames."""

    result_df = pd.merge(df1.df, df2.df, how="cross")

    return NamedDataFrame(expr["table_alias"], result_df)

# TODO: finish implementing semi joins and polish 
def exec_join(expr, df1, df2):
    """Execute the given join operation on the given DataFrames."""

    join_type = expr["join_type"]
    condition = expr.get("condition")
    attributes = expr.get("attributes")

    df1_name = df1.origin_name
    df2_name = df2.origin_name

    if "semi" in join_type:
        raise ValueError("semi not supported yet")

    if condition is None:
        result_df = pd.merge(df1.df, df2.df, how=join_type, on=attributes, suffixes=(f'_{df1_name}', f'_{df2_name}'))

    else:
        cross = pd.merge(df1.df, df2.df, how="cross", suffixes=(f'_{df1_name}', f'_{df2_name}'))
        mask = evaluate_comparison_cond(cross, condition)

        result_df = cross[mask]

    return NamedDataFrame(expr["table_alias"], result_df)

def exec_divide(expr, df1, df2):
    """Execute the given divide operation on the given DataFrames."""

    df1 = df1.df.drop_duplicates()
    df2 = df2.df.drop_duplicates()

    # divisor columns and columns separate from the divisors
    divisor_cols = df2.columns.tolist()
    remaining_cols = [col for col in df1.columns if col not in divisor_cols]

    # ensure df2 columns are subset of df1
    if not set(divisor_cols).issubset(set(df1.columns)):
        raise ValueError("All columns in divisor must exist in dividend.")

    # get all combinations of the canidates with the divisors
    candidates = df1[remaining_cols].drop_duplicates()
    cross = candidates.merge(df2, how='cross')

    # find which tuples in df1 match the candidate combinations
    matched = df1.merge(cross)

    # count how many divisor matches per candidate group
    grouped = matched.groupby(remaining_cols, sort=False)
    full_matches = grouped.size().reset_index(name='_count')

    # only keep those with a full match to all df2 rows
    result_df = (full_matches[full_matches['_count'] == len(df2)])[remaining_cols]

    return NamedDataFrame(expr["table_alias"], result_df)

## ~~~~~~~~ TABLES, ATTRIBUTES, & OTHER ~~~~~~~~ ##

def resolve_operand(df, operand):
    """Resolve a column reference, literal, or math expression into a value or Series."""

    try:

        # evaltuate math or comp conditions
        if isinstance(operand, dict):
            t = operand.get("type")
            if t == "math_cond":
                return evaluate_math_cond(df, operand)
            elif t == "comp_cond":
                return evaluate_comparison_cond(df, operand)
            else:
                raise ValueError(f"Unknown operand type: {t}")

        # resolve table.attr names
        if isinstance(operand, list):
            join_name = operand[-1] + "_" + operand[0]
            if operand[-1] in df:
                return df[operand[-1]]
            elif join_name in df:
                return df[join_name]

        # if a string wrapped in "", assume a string literal, otherwise col name
        if isinstance(operand, str):
            if ((operand.startswith('"') and operand.endswith('"')) 
                or (operand.startswith("'") and operand.endswith("'"))):
                return operand.strip('"\'')
            return df[operand]

        # if a number, just return
        return operand

    except KeyError as e:
        raise InvalidColumnName(e.args[0])

def evaluate_math_cond(df, expr):
    """Evaluate a math expression with the given df."""
    left = resolve_operand(df, expr["left"])
    right = resolve_operand(df, expr["right"])
    op = expr["op"]

    if op == "+":
        return left + right
    elif op == "-":
        return left - right
    elif op == "*":
        return left * right
    elif op == "/":
        return left / right
    elif op == "%":
        return left % right
    elif op == "^":
        return left ** right
    else:
        raise ValueError(f"Unsupported math operator: {op}")
    
def evaluate_comparison_cond(df, cond):
    """Recursively evaluate a comparison or logical condition."""

    # recursive and/or condition
    if cond["op"] in {"AND", "OR"}:
        left_mask = evaluate_comparison_cond(df, cond["left"])
        right_mask = evaluate_comparison_cond(df, cond["right"])

        if cond["op"] == "AND":
            return left_mask & right_mask
        else:
            return left_mask | right_mask

    # binary comparison operator
    left_val = resolve_operand(df, cond["left"])
    right_val = resolve_operand(df, cond["right"])
    op = cond["op"]

    if op == ">":
        return left_val > right_val
    elif op == "<":
        return left_val < right_val
    elif op in {"=", "=="}:
        return left_val == right_val
    elif op == ">=":
        return left_val >= right_val
    elif op == "<=":
        return left_val <= right_val
    elif op == "!=":
        return left_val != right_val
    else:
        raise ValueError(f"Unsupported comparison operator: {op}")

def process_attributes(df, attributes_expr):
    """
    Process a list of attribute expressions, modifying the DataFrame in place
    to include any computed columns, and returning a list of column names.

    Args:
        df (pd.DataFrame): the dataframe to operate on
        attributes_expr (list): list of attributes (strings, lists, or dicts)

    Returns:
        list: list of column names (including computed ones)
    """
    processed_attrs = []

    for attr in attributes_expr:
        if isinstance(attr, dict):
            # compute an aliased column
            alias = attr["alias"]
            df[alias] = evaluate_math_cond(df, attr["cond"])
            processed_attrs.append(alias)
        elif isinstance(attr, list):
            # handle a dotted column name
            processed_attrs.append(attr[-1])
        else:
            processed_attrs.append(attr)

    return processed_attrs

def parse_aggr_conds(aggr_conds):
    """
    Convert aggr_cond expressions into a dict of output_column -> (col, agg_func),
    e.g., {"sum_salary": ("salary", "sum")}
    """
    aggr_funcs = {}
    AGGR_OP_MAP = {
        "sum": "sum",
        "count": "count",
        "avg": "mean",
        "min": "min",
        "max": "max"
    }

    for aggr in aggr_conds:
        op = aggr["aggr"].lower()
        attr = aggr["attr"]
        col = attr[-1] if isinstance(attr, list) else attr

        # Special handling for count(*)
        if op == "count" and col == "*":
            alias = "count_star"
            aggr_funcs[alias] = ("*", "size")  # pandas .size() handles count(*)
        else:
            if op not in AGGR_OP_MAP:
                raise ValueError(f"Unsupported aggregation operator: {op}")
            alias = f"{op}_{col}"
            aggr_funcs[alias] = (col, AGGR_OP_MAP[op])

    return aggr_funcs
