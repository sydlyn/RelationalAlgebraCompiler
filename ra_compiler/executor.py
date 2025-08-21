# ra_compiler/executor.py
'''Execute the given translated query and produce a result DF.'''

import pandas as pd
from .mysql import run_query
from .utils import print_error, print_warning
from .exceptions import TableNotFoundError, TableAlreadyExists, InvalidColumnName

saved_results = {}
#TODO: split this file into more smaller files

class NamedDataFrame():
    """
    Represents a pandas DataFrame wiht an associated name.
    
    Attributes:
        name (str): the name of the dataframe  
        df (pd.DataFrame): the corresponding pandas DataFrame  
    """

    def __init__(self, name, df: pd.DataFrame):
        self.name = name
        self.df = df

    def __repr__(self):
        return f"NamedDataFrame(name={self.name}, shape={self.df.shape})"

    def __str__(self):
        return f"NamedDataFrame: {self.name} (shape: {self.df.shape})"

    def copy(self):
        '''Return a copy of the NamedDataFrame.'''
        return NamedDataFrame(self.name, self.df.copy())

def execute(expr):
    '''Execute a given relational algebra query and return a NamedDataFrame.'''

    if not expr:
        raise ValueError("RA expression cannot be empty.")

    operation = None

    try:
        if not isinstance(expr, dict):
            # if it is a string, assume it is a table name and load the table
            if isinstance(expr, str):
                return load_table(expr)

            # otherwise, if it is not a dict through an error
            raise TypeError("RA expression must be a dictionary representing the query.")

        # ensure the expression has an operation type
        operation = expr.get("operation", None)
        if operation is None:
            raise ValueError("RA expression must contain 'operation' key.")

        # recursively get/evaluate the relevant DataFrame(s) for the operation
        ndf, ndf2 = get_tables(expr)

        # execute the operation
        match operation:
            case "list":
                result = exec_list()

            # unary ops
            case "projection":
                result = exec_projection(expr, ndf)
            case "selection":
                result = exec_selection(expr, ndf)
            case "group":
                result = exec_group(expr, ndf)
            case "rename":
                result = exec_rename(expr, ndf)
            case "remove_duplicates":
                result = exec_remove_duplicates(expr, ndf)
            case "sort":
                result = exec_sort(expr, ndf)

            # set ops
            case "union":
                result = exec_union(expr, ndf, ndf2)
            case "intersection":
                result = exec_interestion(expr, ndf, ndf2)
            case "difference":
                result = exec_difference(expr, ndf, ndf2)

            # merge ops
            case "cross":
                result = exec_cross(expr, ndf, ndf2)
            case "join":
                result = exec_join(expr, ndf, ndf2)
            case "divide":
                result = exec_divide(expr, ndf, ndf2)

            case _:
                raise ValueError(f"Operation not supported: {expr['operation']}")

        return result

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

    # ensure the expression has an op type
    op_type = expr.get("op_type", None)
    if op_type is None:
        raise ValueError("RA expression must contain 'op_type' key.")

    match op_type:
        # if a list op, return nothing
        case "list":
            return None, None

        # if a unary op, get and return the one table
        case "unary":
            ndf = load_table(expr.get("table"))
            return ndf, None

        # if a set op, get and check the two tables
        case "set":
            ndf = load_table(expr.get("table1"))
            ndf2 = load_table(expr.get("table2"))

            # ensure the two tables have the same columns
            if not ndf.df.columns.equals(ndf2.df.columns):
                raise ValueError("Set operations require both tables to " \
                                 "have the same columns in the same order.")

            return ndf, ndf2

        # if a merge op, get and return the two tables
        case "merge":
            ndf = load_table(expr.get("table1"))
            ndf2 = load_table(expr.get("table2"))
            return ndf, ndf2

        # otherwise, error
        case _:
            raise ValueError(f"Unsupported operation type: {op_type}")

def load_table(table_name):
    """Return a pandas DataFrame of the desired table from the SQL connection."""

    if table_name is None:
        raise TableNotFoundError()

    # if the table is a string...
    if isinstance(table_name, str):

        # check if a saved result
        if table_name in saved_results:
            return saved_results[table_name].copy()

        # or check if it is a table from SQL connection
        if not check_sql_for_table(table_name):
            raise TableNotFoundError(table_name)

        # if the table exists, load and save it
        query = f"SELECT * FROM `{table_name}`"
        cols, rows = run_query(query)

        # set all NULL values to pd.NA
        cleaned_rows = []
        for row in rows:
            cleaned_rows += [[pd.NA if x is None else x for x in row]]

        df = pd.DataFrame(cleaned_rows, columns=cols)

        ndf = NamedDataFrame(table_name, df)
        saved_results[table_name] = ndf

        return ndf.copy()

    # otherwise, execute the nested operation
    else:
        ndf = execute(table_name)
        return ndf

def check_sql_for_table(table_name):
    """Return True if the given table name is in the connected sql database."""

    check_query = f"SHOW TABLES LIKE '{table_name}'"
    result = run_query(check_query)

    if len(result[1]) == 0:
        return False

    return True

def exec_list():
    """Return a dict of all available tables."""
    tables = {}

    # get all tables from the sql connection
    query = "SHOW TABLES;"
    _, rows = run_query(query)
    for row in rows:
        tables[row[0]] = True

    # get all new tables that have since been saved
    for ndf in saved_results:
        if ndf not in tables:
            tables[ndf] = True

    return list(tables)


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

    return NamedDataFrame(expr['table_alias'], result_df)

def exec_selection(expr, ndf):
    """Execute the given selection operation on the given DataFrame."""

    df = ndf.df
    cond = expr["condition"]
    mask = evaluate_comparison_cond(df, cond)

    if isinstance(mask, bool):
        # if the condition is a boolean, return the original DataFrame
        mask = pd.Series(mask, index=df.index)

    return NamedDataFrame(expr['table_alias'], df[mask])

def exec_group(expr, ndf):
    """Execute the given group operation on the given DataFrame."""

    df = ndf.df
    group_attrs = process_attributes(df, expr["attributes"])
    aggr_funcs = parse_aggr_conds(expr["aggr_cond"], df)

    group_df = df.groupby(group_attrs, sort=False)

    # construct aggregation mapping for pandas
    agg_dict = {}
    for alias, (col, func) in aggr_funcs.items():

        # when count by *, return how many rows are in each group
        if func == "size" and col == "*":
            temp_col = "__count_star_temp"
            df[temp_col] = 1
            agg_dict[alias] = pd.NamedAgg(column=temp_col, aggfunc=func)

        elif func == "count":
            # count when EITHER is not null
            mask = ~pd.concat([df[c].isna() for c in col], axis=1).all(axis=1)

            temp_col = f"__mask_{alias}"
            df[temp_col] = mask.astype(int)

            group_df = df.groupby(group_attrs, sort=False)
            agg_dict[alias] = pd.NamedAgg(column=temp_col, aggfunc="sum")

        else:
            if func == "sum":
                agg_func = lambda x: x.sum() if x.notna().any() else pd.NA
            elif func == "mean":
                agg_func = lambda x: x.mean() if x.notna().any() else pd.NA
            elif func == "min":
                agg_func = lambda x: x.min() if x.notna().any() else pd.NA
            elif func == "max":
                agg_func = lambda x: x.max() if x.notna().any() else pd.NA
            else:
                raise ValueError(f"Unsupported aggregation function: {func}")

            agg_dict[alias] = pd.NamedAgg(column=col, aggfunc=agg_func)

    result_df = group_df.agg(**agg_dict).reset_index()
    return NamedDataFrame(expr['table_alias'], result_df)

def exec_rename(expr, ndf):
    """Rename the table to the given alias."""

    alias = expr.get("table_alias")

    # check if the alias is in the sql database or is a saved table
    if check_sql_for_table(alias) or alias in saved_results:
        raise TableAlreadyExists(alias)

    return NamedDataFrame(alias, ndf.df)

def exec_remove_duplicates(expr, ndf):
    """Remove duplicates from the given DataFrame."""

    return NamedDataFrame(expr["table_alias"], ndf.df.drop_duplicates())

def exec_sort(expr, ndf):
    """Sort the DataFrame based on the given attributes."""

    df = ndf.df
    sort_attrs = expr["sort_attributes"]

    # sort in reverse so the first listed attr has the higher sort importance
    for attr in reversed(sort_attrs):
        # ensure the attribute is a valid column in the DataFrame
        col = attr[0]
        if col not in df.columns:
            raise InvalidColumnName(col)

        df = df.sort_values(by=col, ascending=attr[1], na_position="first")

    return NamedDataFrame(expr['table_alias'], df)


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
    inter = pd.merge(df1_dr, df2_dr, how="inner", on=None, validate="m:m")
    result_df = inter[df1.df.columns]

    return NamedDataFrame(expr['table_alias'], result_df)

def exec_difference(expr, df1, df2):
    """Execute the given difference operation on the given DataFrames."""

    keep_dups = expr.get("keep_dups", False)
    df1_dr, df2_dr = prepare_for_set_op(df1.df, df2.df, keep_dups)

    # merge the data frames and keep only the relevant columns/rows
    merged = pd.merge(df1_dr, df2_dr, how="left", on=None, validate="m:m",
                      indicator=True)
    diff = merged[merged['_merge'] == 'left_only']
    result_df = diff[df1.df.columns]

    return NamedDataFrame(expr["table_alias"], result_df)

def prepare_for_set_op(df1, df2, keep_dups=False):
    """Prepares two DataFrames for set operations by either 
    dropping duplicates or adding row counts (for bag semantics)."""

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


## ~~~~~~~~ MERGE OPERATIONS ~~~~~~~~ ##

def prepare_for_merge_op(left_df, right_df):
    """Ensure no duplicate column names between left_df and right_df.
    Adds _L or _R suffixes to duplicates, preserving any existing suffixes."""

    # grab the original columns from each table
    left_cols = left_df.columns.tolist()
    right_cols = right_df.columns.tolist()

    def strip_suffixes(col):
        # strip all trailing _L or _R occurrences
        while col.endswith(("_L", "_R")):
            col = col[:-2]
        return col

    # remove any existing _L/_R suffixes to get the base column name
    left_bases = [strip_suffixes(c) for c in left_cols]
    right_bases = [strip_suffixes(c) for c in right_cols]

    duplicate_cols = set(left_bases) & set(right_bases)

    # for the left table, add a _L subscript for columns that have a duplicate
    for i, col in enumerate(left_bases):
        if col in duplicate_cols:
            left_cols[i] += "_L"

    # for the right table, add a _L subscript for columns that have a duplicate
    for i, col in enumerate(right_bases):
        if col in duplicate_cols:
            right_cols[i] += "_R"

    # reset the df columns to be the updated column names
    left_df.columns = left_cols
    right_df.columns = right_cols

def exec_cross(expr, ndf1, ndf2):
    """Execute the given cross operation on the given DataFrames."""

    df1 = ndf1.df
    df2 = ndf2.df

    prepare_for_merge_op(df1, df2)
    result_df = pd.merge(df1, df2, how="cross",
                         suffixes=('_L', '_R'), validate="m:m")

    return NamedDataFrame(expr["table_alias"], result_df)


def exec_join(expr, ndf1, ndf2):
    """Execute the given join operation on the given DataFrames."""

    join_type = expr["join_type"]
    condition = expr.get("condition")
    attributes = expr.get("attributes")
    alias = expr.get("table_alias")

    # add helper ids to the DataFrames and grab all the columns
    df1_dr = ndf1.df.reset_index().rename(columns={'index': '_left_id'})
    df2_dr = ndf2.df.reset_index().rename(columns={'index': '_right_id'})
    orig_cols = df1_dr.columns.tolist() + df2_dr.columns.tolist()

    try:
        # if no condition is specified, use a cross join, otherwise use the specified join
        if condition:
            merge_how = 'cross'
        elif 'semi' in join_type:
            merge_how = 'inner'
        else:
            merge_how = join_type

        merge = pd.merge(df1_dr, df2_dr, how=merge_how, on=attributes,
                         suffixes=('_L', '_R'), validate="m:m")
    except KeyError as e:
        raise InvalidColumnName(e.args[0]) from e

    # get the columns that are from the left and the right dataframes after the join
    left_cols = [c for c in merge.columns if (c in df1_dr.columns or c.endswith('_L'))]
    right_cols = [c for c in merge.columns if (c in df2_dr.columns or c.endswith('_R'))]

    # if no condition is specified, merge is the result of panda's specialized merge
    if not condition:
        result_df = handle_output_cols(merge, join_type, left_cols, right_cols, orig_cols)
        return clean_join_result(alias, result_df)

    # if a condition is specified, mask the crossed DataFrame
    mask = evaluate_comparison_cond(merge, condition)
    if isinstance(mask, bool):
        # if the condition is a boolean, return the original DataFrame
        mask = pd.Series(condition, index=merge.index)

    masked = merge[mask]

    checked_dups = handle_output_cols(masked, join_type, left_cols, right_cols, orig_cols)

    if "semi" in join_type:
        return clean_join_result(alias, checked_dups)
    elif "inner" in join_type:
        return clean_join_result(alias, masked)
    else:
        outer_result =  handle_outer_join(masked, merge, join_type, left_cols, right_cols)
        return clean_join_result(alias, outer_result)

def handle_output_cols(merge, join_type, left_cols, right_cols, original_cols):
    """Handle duplicate columns of merged df."""

    # if a semi-join, only keep the left or right columns
    if "semi" in join_type:
        if "left" in join_type:
            result_df = (
                merge[left_cols].rename(columns=lambda x: x.replace('_L', ''))
                    .drop_duplicates('_left_id')
            )
        else:
            result_df = (
                merge[right_cols].rename(columns=lambda x: x.replace('_R', ''))
                    .drop_duplicates('_right_id')
            )
    else:
        # print a warning if the output has duplicate columns
        if set(merge.columns) != set(original_cols):
            print_warning("Duplicate columns found in join." \
                          "Can cause unexpected results." \
                          "Please ensure unique column names across tables.",
                          "JoinWarning")
        result_df = merge

    return result_df

def handle_outer_join(masked, merge, join_type, left_cols, right_cols):
    """Handle outer join output. Fill unmatched rows with NaN."""

    # get the original rows that matched at least once
    matched_left = set(masked['_left_id'])
    matched_right = set(masked['_right_id'])

    # get the rows that do not have a match from the left and right DataFrames
    left_only = (
        merge.loc[~merge['_left_id'].isin(matched_left), left_cols]
            .drop_duplicates('_left_id')
            .copy()
    )
    right_only = (
        merge.loc[~merge['_right_id'].isin(matched_right), right_cols]
            .drop_duplicates('_right_id')
            .copy()
    )

    # fill the unmatched rows with NaN values for the other side
    for rc in right_cols:
        left_only[rc] = pd.NA

    for lc in left_cols:
        right_only[lc] = pd.NA

    if "left" in join_type:
        parts = [masked, left_only]
    elif "right" in join_type:
        parts = [masked, right_only]
    elif "outer" in join_type:
        parts = [masked, left_only, right_only]
    else:
        raise ValueError("Invalid join type given.")

    # align and combine
    out = pd.concat(parts, ignore_index=True)
    out = out.convert_dtypes()

    if "left" in join_type:
        out = out.sort_values(by='_left_id')
    elif "right" in join_type:
        out = out.sort_values(by='_right_id')

    return out

def clean_join_result(alias, result_df):

    # drop helper ids
    if '_left_id' in result_df.columns:
        result_df = result_df.drop(columns=['_left_id'])
    if '_right_id' in result_df.columns:
        result_df = result_df.drop(columns=['_right_id'])

    # reset the index
    result_df.reset_index(drop=True, inplace=True)
    result_df = result_df.convert_dtypes()

    return NamedDataFrame(alias, result_df)

def exec_divide(expr, ndf1, ndf2):
    """Execute the given divide operation on the given DataFrames."""

    df1 = ndf1.df.drop_duplicates()
    df2 = ndf2.df.drop_duplicates()

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

def resolve_operand(df, operand, left=True):
    """Resolve a column reference, literal, or math expression into a value or Series."""

    try:

        # evaltuate math or comp conditions
        if isinstance(operand, dict):
            t = operand.get("type")
            if t == "math_cond":
                return evaluate_math_cond(df, operand)

            if t == "comp_cond":
                return evaluate_comparison_cond(df, operand)
            if t == "alias":
                alias = operand["alias"]
                df[alias] = resolve_operand(df, operand["attr"])
                return df[alias]

            raise ValueError(f"Unknown operand type: {t}")

        # resolve table.attr names
        if isinstance(operand, list):
            join_name = operand[-1] + "_L" if left else operand[-1] + "_R"
            if operand[-1] in df:
                return df[operand[-1]]

            if join_name in df:
                return df[join_name]

        if isinstance(operand, str):
            # if a string wrapped in "", assume a string literal
            if ((operand.startswith('"') and operand.endswith('"'))
                or (operand.startswith("'") and operand.endswith("'"))):
                return operand.strip('"\'')

            # if a string is a column name, return the column
            join_name = operand + "_L" if left else operand + "_R"
            if join_name in df:
                return df[join_name]
            return df[operand]

        # if a number, just return
        return operand

    except KeyError as e:
        raise InvalidColumnName(e.args[0]) from e

def evaluate_math_cond(df, expr):
    """Evaluate a math expression with the given df."""
    left = resolve_operand(df, expr["left"])
    right = resolve_operand(df, expr["right"], left=False)
    op = expr["op"]

    match op:
        case "+":
            return left + right
        case "-":
            return left - right
        case "*":
            return left * right
        case "/":
            return left / right
        case "%":
            return left % right
        case "^":
            return left ** right
        case _:
            raise ValueError(f"Unsupported math operator: {op}")

def evaluate_comparison_cond(df, cond):
    """Recursively evaluate a comparison or logical condition."""

    # if the condition is a boolean, return a full/empty mask
    if isinstance(cond, bool):
        return pd.Series(cond, index=df.index, dtype="boolean")

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
    right_val = resolve_operand(df, cond["right"], left=False)
    op = cond["op"]

    if not isinstance(left_val, pd.Series):
        left_val = pd.Series([left_val] * len(df), index=df.index)
    if not isinstance(right_val, pd.Series):
        right_val = pd.Series([right_val] * len(df), index=df.index)

    # get which rows have NA's
    na_mask = left_val.isna() | right_val.isna()
    non_na_idx = ~na_mask

    # default rows with NA's to False
    result = pd.Series(False, index=df.index, dtype='boolean')

    if op == ">":
        result[non_na_idx] = left_val[non_na_idx] > right_val[non_na_idx]
    elif op == "<":
        result[non_na_idx] = left_val[non_na_idx] < right_val[non_na_idx]
    elif op in {"=", "=="}:
        result[non_na_idx] = left_val[non_na_idx] == right_val[non_na_idx]
    elif op == ">=":
        result[non_na_idx] = left_val[non_na_idx] >= right_val[non_na_idx]
    elif op == "<=":
        result[non_na_idx] = left_val[non_na_idx] <= right_val[non_na_idx]
    elif op == "!=":
        result[non_na_idx] = left_val[non_na_idx] != right_val[non_na_idx]
    else:
        raise ValueError(f"Unsupported comparison operator: {op}")

    return result

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
            if "cond" in attr:
                df[alias] = evaluate_math_cond(df, attr["cond"])
            elif "attr" in attr:
                df[alias] = resolve_operand(df, attr["attr"])
            processed_attrs.append(alias)
        elif isinstance(attr, list):
            # handle a dotted column name
            processed_attrs.append(attr[-1])
        else:
            if attr == "*":
                # if the attribute is *, return all columns
                processed_attrs = df.columns.tolist()
            else:
                # otherwise, just add the column name
                processed_attrs.append(attr)

    return processed_attrs

def parse_aggr_conds(aggr_conds, df):
    """Convert aggr_cond expressions into a dict of output_column -> (col, agg_func)"""
    aggr_funcs = {}
    for aggr in aggr_conds:
        # grab the alias if given one
        if isinstance(aggr, list):
            alias = aggr[1]
            aggr = aggr[0]
        else:
            alias = None

        op = aggr["aggr"].lower()
        attr = aggr.get("attr", "*")
        attrs = process_attributes(df, attr)

        # Special handling for count(*)
        if op == "count" and attr == ["*"]:
            alias = alias if alias else "count_star"
            aggr_funcs[alias] = ("*", "size")
        else:
            alias = alias if alias else f"{op}_{'_'.join(attrs)}"
            aggr_funcs[alias] = (attr, op)

    return aggr_funcs
