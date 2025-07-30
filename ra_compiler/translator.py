# ra_compiler/translator.py

from lark import Transformer
from .utils import clean_exit, print_debug

class RATranslator(Transformer):
    """Transforms a Lark parse tree into a relational algebra representation."""

    def __init__(self, query_count):
        super().__init__()
        self.alias_counter = 0
        self.query_count = query_count

    def _new_alias(self):
        alias = f"_rac_q{self.query_count}_t{self.alias_counter}"
        self.alias_counter += 1
        return alias
    
    def add_alias(self, expr):
        if "table_alias" not in expr:
            expr["table_alias"] = self._new_alias()
        return expr

    dup_indicator = "^d"

    def start(self, items):
        return items[0]

    ## ~~~~~~~~ UNARY OPERATIONS ~~~~~~~~ ##

    def unary_ops(self, items):
        return { "op_type": "unary"} | items[0]
    
    def projection(self, items):
        prefix, attributes, table = items

        return self.add_alias({
            "operation": "projection",
            "keep_dups": self.dup_indicator in prefix.value,
            "table": table,
            "attributes": attributes,
        })

    def selection(self, items):
        condition, table = items

        return self.add_alias({
            "operation": "selection",
            "table": table,
            "condition": condition,
        })
    
    def group(self, items):
        attributes, aggr_cond, table = items
        return self.add_alias({
            "operation": "group",
            "table": table,
            "attributes": attributes,
            "aggr_cond": aggr_cond,
        })
    
    def rename(self, items):
        cname, table = items
        return {
            "operation": "rename",
            "table": table,
            "table_alias": cname,
        }

    def remove_duplicates(self, items):
        table = items[0]
        return self.add_alias({
            "operation": "remove_duplicates",
            "table": table,
        })
    
    def sort(self, items):
        sort_attributes, table = items
        return self.add_alias({
            "operation": "sort",
            "table": table,
            "sort_attributes": sort_attributes,
        })
    
    ## ~~~~~~~~ SET OPERATIONS ~~~~~~~~ ##

    def set_ops(self, items):
        return { "op_type": "set"} | items[0]

    def union(self, items):
        table1, prefix, table2 = items
        return self.add_alias({
            "operation": "union",
            "keep_dups": True if self.dup_indicator in prefix.value else False,
            "table1": table1,
            "table2": table2,
        })
    
    def intersection(self, items):
        table1, prefix, table2 = items
        return self.add_alias({
            "operation": "intersection",
            "keep_dups": True if self.dup_indicator in prefix.value else False,
            "table1": table1,
            "table2": table2,
        })
    
    def difference(self, items):
        table1, prefix, table2 = items
        return self.add_alias({
            "operation": "difference",
            "keep_dups": True if self.dup_indicator in prefix.value else False,
            "table1": table1,
            "table2": table2,
        })

    ## ~~~~~~~~ JOIN OPERATIONS ~~~~~~~~ ##

    def join_ops(self, items):
        return { "op_type": "join"} | items[0]

    def cross(self, items):
        table1, table2 = items
        return self.add_alias({
            "operation": "cross",
            "table1": table1,
            "table2": table2,
        })
    
    def join(self, items):
        if len(items) == 3:
            table1, join_prefix, table2 = items
            specs = None
        elif len(items) == 4:
            table1, join_prefix, specs, table2 = items                
        else:
            raise ValueError(f"Unexpected number of items in join: {len(items)}")
        
        # determine what type of join by the prefix
        def get_join_type(join_prefix):
            join = join_prefix.lstrip('/').lower()

            if "semi" in join:
                if "right" in join:
                    return "semi", "right"
                else:
                    return "semi", "left"
                
            if "right" in join:
                return "right"
            elif "left" in join:
                return "left"
            elif "full" in join or "outer" in join:
                return "outer"
            else:
                return "inner"

        join_dict = {
            "operation": "join",
            "join_type": get_join_type(join_prefix),
            "table1": table1,
            "table2": table2,
        }

        # deterine if given a list of attributes or a comp condition
        if isinstance(specs, list):
            join_dict["attributes"] = specs
        elif isinstance(specs, dict):
            join_dict["condition"] = specs

        return self.add_alias(join_dict)

    def divide(self, items):
        table1, table2 = items
        return self.add_alias({
            "operation": "divide",
            "table1": table1,
            "table2": table2,
        })

    ## ~~~~~~~~ TABLES, ATTRIBUTES, & OTHER ~~~~~~~~ ##

    def table(self, items):
        return items[0]

    def attributes(self, attrs):
        return attrs
    
    def sort_attributes(self, attrs):
        return attrs
            
    #TODO: implemet b->b_new
    def attr(self, items):
        if isinstance(items[0], (dict)):
            return {
                'alias': str(items[1]),
                'cond': items[0]
            }
        else:
            return items
        
    def sort_attr(self, items):
        return (items[0], items[1])
    
    def math_cond(self, items):
        if len(items) == 1:
            return items[0]
        
        left, op, right = items
        return {
            "type": "math_cond", 
            "left": left, 
            "op": op, 
            "right": right
        }

    def comp_cond(self, items):
        if len(items) == 1:
            return items[0]
        
        left, op, right = items
        return {
            "type": "comp_cond", 
            "left": left, 
            "op": op, 
            "right": right
        }
    
    def aggr_conds(self, items):
        return items
    
    def aggr_func(self, items):
        if len(items) == 1:
            return {"aggr": items[0]}
        elif len(items) == 2:
            return {"aggr": items[0], "attr": items[1]}
        else:
            raise ValueError("Invalid aggregation condition format")
        
    def MATH_OP(self, token):
        return token.value
    
    def COMP_OP(self, token):
        return token.value
    
    def AGGR_OP(self, token):
        return token.value
    
    def AND(self, _):
        return "AND"
    
    def OR(self, _):
        return "OR"
    
    def SORT_DIR(self, token):
        return token.value.upper() == "ASC"
    
    def CNAME(self, token):
        return token.value
    
    def NUMBER(self, token):
        return int(token.value)
    
    def ESCAPED_STRING(self, token):
        return token.value
