# ra_compiler/translator.py

from lark import Transformer
from .utils import clean_exit

class RATranslator(Transformer):
    """Transforms a Lark parse tree into a relational algebra representation."""

    try: 
        dup_indicator = "^d"

        def start(self, args):
            return args[0]

        ## ~~~~~~~~ UNARY OPERATIONS ~~~~~~~~ ##

        def unary_ops(self, args):
            return { "op_type": "unary"} | args[0]
        
        def projection(self, args):
            prefix, attributes, table = args

            return {
                "operation": "projection",
                "keep_dups": self.dup_indicator in prefix.value,
                "table": table,
                "attributes": attributes,
            }

        def selection(self, args):
            condition, table = args

            return {
                "operation": "selection",
                "table": table,
                "condition": condition,
            }
        
        def group(self, args):
            attributes, aggr_cond, table = args
            return {
                "operation": "group",
                "table": table,
                "attributes": attributes,
                "aggr_cond": aggr_cond,
            }
        
        def rename(self, args):
            cname, table = args
            return {
                "operation": "rename",
                "table": table,
                "alias": cname,
            }
    
        def remove_duplicates(self, args):
            table = args[0]
            return {
                "operation": "remove_duplicates",
                "table": table,
            }
        
        def sort(self, args):
            sort_attributes, table = args
            return {
                "operation": "sort",
                "table": table,
                "sort_attributes": sort_attributes,
            }
        
        ## ~~~~~~~~ SET OPERATIONS ~~~~~~~~ ##

        def set_ops(self, args):
            return { "op_type": "set"} | args[0]

        def union(self, args):
            table1, prefix, table2 = args
            return {
                "operation": "union",
                "keep_dups": True if self.dup_indicator in prefix.value else False,
                "table1": table1,
                "table2": table2,
            }
        
        def intersection(self, args):
            table1, prefix, table2 = args
            return {
                "operation": "intersection",
                "keep_dups": True if self.dup_indicator in prefix.value else False,
                "table1": table1,
                "table2": table2,
            }
        
        def difference(self, args):
            table1, prefix, table2 = args
            return {
                "operation": "difference",
                "keep_dups": True if self.dup_indicator in prefix.value else False,
                "table1": table1,
                "table2": table2,
            }

        ## ~~~~~~~~ JOIN OPERATIONS ~~~~~~~~ ##

        def join_ops(self, args):
            return { "op_type": "join"} | args[0]

        def cross(self, args):
            table1, table2 = args
            return {
                "operation": "cross",
                "table1": table1,
                "table2": table2,
            }
        
        def join(self, items):
            if len(items) == 3:
                left, join_type, right = items
                condition = None
            elif len(items) == 4:
                left, join_type, condition, right = items
            else:
                raise ValueError(f"Unexpected number of items in join: {len(items)}")

            return {
                "operation": "join",
                "join_type": str(join_type),
                "left": left,
                "right": right,
                "condition": condition
            }

        def join(self, items):
            if len(items) == 3:
                left, join_type, right = items
                condition = None
            elif len(items) == 4:
                left, join_type, condition, right = items
            else:
                raise ValueError(f"Unexpected number of items in join: {len(items)}")

            # Normalize join type string
            join_str = str(join_type).lower().replace('/', '').replace('_', ' ')
            if 'semi' in join_str:
                join_sql = 'SEMI JOIN'
            elif 'inner' in join_str:
                join_sql = 'INNER JOIN'
            elif 'left' in join_str:
                join_sql = 'LEFT OUTER JOIN'
            elif 'right' in join_str:
                join_sql = 'RIGHT OUTER JOIN'
            elif 'full' in join_str:
                join_sql = 'FULL OUTER JOIN'
            else:
                join_sql = 'JOIN'  # fallback

            # Determine ON condition
            if condition is None:
                on_clause = ""
            elif isinstance(condition, list):  # case: attribute list e.g. {A,B}
                # Produce C1.A = C2.A AND C1.B = C2.B
                conditions = [f"{left}.{attr} = {right}.{attr}" for attr in condition]
                on_clause = " ON " + " AND ".join(conditions)
            else:  # case: a specific comparison like C1.A = C2.B
                on_clause = f" ON {condition}"

            return {"SQL": f"SELECT * FROM {left} {join_sql} {right}{on_clause}"}

        def divide(self, args):
            table1, table2 = args
            return {
                "operation": "divide",
                "table1": table1,
                "table2": table2,
            }

        ## ~~~~~~~~ TABLES, ATTRIBUTES, & OTHER ~~~~~~~~ ##

        def table(self, args):
            return args[0]

        def attributes(self, attrs):
            return attrs
            
        def attr(self, args):
            return {
                'alias': str(args[1]),
                'cond': args[0]
            }
        
        def math_cond(self, args):
            left, op, right = args
            eq = "" + left + op + str(right) + ""
            return {"left": left, "op": op, "right": right, "eq": eq}

        def comp_cond(self, args):
            left, op, right = args
            eq = "" + left + op + str(right) + ""
            return {"left": left, "op": op, "right": right, "eq": eq}

        def aggr_cond(self, args):
            if len(args) == 1:
                return {"aggr": str(args[0])}
            elif len(args) == 2:
                return {"aggr": str(args[0]), "alias": str(args[1])}
            else:
                raise ValueError("Invalid aggregation condition format")
            
        def MATH_OP(self, token):
            return token.value
        
        def COMP_OP(self, token):
            return token.value
        
        def AND(self, _):
            return "AND"
        
        def CNAME(self, token):
            return token.value
        
        def NUMBER(self, token):
            return int(token.value)
    
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        clean_exit(1)
