"""
Dynamic query builder for SQL databases.

Provides functionality to build SQL queries dynamically for PostgreSQL and MySQL
with support for SELECT, JOIN, WHERE, ORDER BY, and other SQL constructs.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class JoinSpec:
    """Specification for table joins."""
    table: str
    join_type: str = 'INNER'  # INNER, LEFT, RIGHT, FULL
    on_condition: str = ''
    alias: Optional[str] = None


class QueryBuilder:
    """Dynamic SQL query builder for PostgreSQL and MySQL."""
    
    def __init__(self, database_type: str):
        """Initialize query builder for specific database type."""
        if database_type not in ['postgresql', 'mysql']:
            raise ValueError(f"Unsupported database type: {database_type}")
        
        self.database_type = database_type
        self.quote_char = '"' if database_type == 'postgresql' else '`'
    
    def build_select_query(self,
                          table: str,
                          columns: Optional[List[str]] = None,
                          where_clause: Optional[str] = None,
                          order_by: Optional[str] = None,
                          group_by: Optional[List[str]] = None,
                          having: Optional[str] = None,
                          limit: Optional[int] = None,
                          offset: Optional[int] = None,
                          distinct: bool = False) -> str:
        """Build SELECT query."""
        
        # SELECT clause
        if columns:
            columns_str = ', '.join([self._quote_identifier(col) for col in columns])
        else:
            columns_str = '*'
        
        select_clause = 'SELECT DISTINCT' if distinct else 'SELECT'
        query = f"{select_clause} {columns_str}"
        
        # FROM clause
        query += f" FROM {self._quote_identifier(table)}"
        
        # WHERE clause
        if where_clause:
            query += f" WHERE {where_clause}"
        
        # GROUP BY clause
        if group_by:
            group_columns = ', '.join([self._quote_identifier(col) for col in group_by])
            query += f" GROUP BY {group_columns}"
        
        # HAVING clause
        if having:
            query += f" HAVING {having}"
        
        # ORDER BY clause
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # LIMIT and OFFSET
        if limit is not None:
            if self.database_type == 'postgresql':
                query += f" LIMIT {limit}"
                if offset is not None and offset > 0:
                    query += f" OFFSET {offset}"
            else:  # mysql
                if offset is not None and offset > 0:
                    query += f" LIMIT {offset}, {limit}"
                else:
                    query += f" LIMIT {limit}"
        
        return query
    
    def build_count_query(self,
                         table: str,
                         where_clause: Optional[str] = None,
                         distinct_column: Optional[str] = None) -> str:
        """Build COUNT query."""
        
        if distinct_column:
            count_expr = f"COUNT(DISTINCT {self._quote_identifier(distinct_column)})"
        else:
            count_expr = "COUNT(*)"
        
        query = f"SELECT {count_expr} FROM {self._quote_identifier(table)}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        return query
    
    def build_join_query(self,
                        main_table: str,
                        join_specs: List[Dict[str, str]],
                        columns: Optional[List[str]] = None,
                        where_clause: Optional[str] = None,
                        order_by: Optional[str] = None,
                        group_by: Optional[List[str]] = None,
                        having: Optional[str] = None,
                        limit: Optional[int] = None) -> str:
        """Build query with table joins."""
        
        # SELECT clause
        if columns:
            columns_str = ', '.join([self._quote_identifier(col) for col in columns])
        else:
            columns_str = '*'
        
        query = f"SELECT {columns_str}"
        
        # FROM clause with main table
        query += f" FROM {self._quote_identifier(main_table)}"
        
        # JOIN clauses
        for join_spec in join_specs:
            join_type = join_spec.get('join_type', 'INNER').upper()
            join_table = join_spec['table']
            on_condition = join_spec['on_condition']
            alias = join_spec.get('alias')
            
            table_ref = self._quote_identifier(join_table)
            if alias:
                table_ref += f" AS {self._quote_identifier(alias)}"
            
            query += f" {join_type} JOIN {table_ref} ON {on_condition}"
        
        # WHERE clause
        if where_clause:
            query += f" WHERE {where_clause}"
        
        # GROUP BY clause
        if group_by:
            group_columns = ', '.join([self._quote_identifier(col) for col in group_by])
            query += f" GROUP BY {group_columns}"
        
        # HAVING clause
        if having:
            query += f" HAVING {having}"
        
        # ORDER BY clause
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # LIMIT
        if limit is not None:
            query += f" LIMIT {limit}"
        
        return query
    
    def build_insert_query(self,
                          table: str,
                          data: Dict[str, Any],
                          on_conflict: Optional[str] = None) -> str:
        """Build INSERT query."""
        
        columns = list(data.keys())
        values = list(data.values())
        
        columns_str = ', '.join([self._quote_identifier(col) for col in columns])
        placeholders = ', '.join(['%s'] * len(values))
        
        query = f"INSERT INTO {self._quote_identifier(table)} ({columns_str}) VALUES ({placeholders})"
        
        # Handle conflicts (PostgreSQL: ON CONFLICT, MySQL: ON DUPLICATE KEY)
        if on_conflict:
            if self.database_type == 'postgresql':
                query += f" ON CONFLICT {on_conflict}"
            else:  # mysql
                query += f" ON DUPLICATE KEY UPDATE {on_conflict}"
        
        return query
    
    def build_update_query(self,
                          table: str,
                          data: Dict[str, Any],
                          where_clause: str) -> str:
        """Build UPDATE query."""
        
        set_clauses = []
        for column in data.keys():
            set_clauses.append(f"{self._quote_identifier(column)} = %s")
        
        set_clause = ', '.join(set_clauses)
        
        query = f"UPDATE {self._quote_identifier(table)} SET {set_clause} WHERE {where_clause}"
        
        return query
    
    def build_delete_query(self,
                          table: str,
                          where_clause: str) -> str:
        """Build DELETE query."""
        
        query = f"DELETE FROM {self._quote_identifier(table)} WHERE {where_clause}"
        
        return query
    
    def build_create_table_query(self,
                                table: str,
                                columns: List[Dict[str, str]],
                                primary_key: Optional[List[str]] = None,
                                foreign_keys: Optional[List[Dict[str, str]]] = None,
                                indexes: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build CREATE TABLE query."""
        
        column_definitions = []
        
        for col in columns:
            col_def = f"{self._quote_identifier(col['name'])} {col['type']}"
            
            if col.get('not_null'):
                col_def += " NOT NULL"
            
            if col.get('default'):
                col_def += f" DEFAULT {col['default']}"
            
            if col.get('unique'):
                col_def += " UNIQUE"
            
            column_definitions.append(col_def)
        
        # Primary key
        if primary_key:
            pk_columns = ', '.join([self._quote_identifier(col) for col in primary_key])
            column_definitions.append(f"PRIMARY KEY ({pk_columns})")
        
        # Foreign keys
        if foreign_keys:
            for fk in foreign_keys:
                fk_def = (
                    f"FOREIGN KEY ({self._quote_identifier(fk['column'])}) "
                    f"REFERENCES {self._quote_identifier(fk['ref_table'])} "
                    f"({self._quote_identifier(fk['ref_column'])})"
                )
                
                if fk.get('on_delete'):
                    fk_def += f" ON DELETE {fk['on_delete']}"
                
                if fk.get('on_update'):
                    fk_def += f" ON UPDATE {fk['on_update']}"
                
                column_definitions.append(fk_def)
        
        columns_clause = ',\n    '.join(column_definitions)
        query = f"CREATE TABLE {self._quote_identifier(table)} (\n    {columns_clause}\n)"
        
        return query
    
    def build_create_index_query(self,
                                index_name: str,
                                table: str,
                                columns: List[str],
                                unique: bool = False,
                                where_clause: Optional[str] = None) -> str:
        """Build CREATE INDEX query."""
        
        index_type = "UNIQUE INDEX" if unique else "INDEX"
        columns_str = ', '.join([self._quote_identifier(col) for col in columns])
        
        query = (
            f"CREATE {index_type} {self._quote_identifier(index_name)} "
            f"ON {self._quote_identifier(table)} ({columns_str})"
        )
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        return query
    
    def build_aggregation_query(self,
                               table: str,
                               aggregations: List[Dict[str, str]],
                               group_by: Optional[List[str]] = None,
                               where_clause: Optional[str] = None,
                               having: Optional[str] = None,
                               order_by: Optional[str] = None) -> str:
        """Build aggregation query with GROUP BY."""
        
        # Build aggregation expressions
        agg_expressions = []
        for agg in aggregations:
            func = agg['function'].upper()  # COUNT, SUM, AVG, MIN, MAX
            column = agg.get('column', '*')
            alias = agg.get('alias')
            
            if column != '*':
                column = self._quote_identifier(column)
            
            expr = f"{func}({column})"
            
            if alias:
                expr += f" AS {self._quote_identifier(alias)}"
            
            agg_expressions.append(expr)
        
        # Add GROUP BY columns to SELECT
        select_columns = []
        if group_by:
            select_columns.extend([self._quote_identifier(col) for col in group_by])
        
        select_columns.extend(agg_expressions)
        columns_str = ', '.join(select_columns)
        
        query = f"SELECT {columns_str} FROM {self._quote_identifier(table)}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if group_by:
            group_columns = ', '.join([self._quote_identifier(col) for col in group_by])
            query += f" GROUP BY {group_columns}"
        
        if having:
            query += f" HAVING {having}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        return query
    
    def build_window_function_query(self,
                                   table: str,
                                   columns: List[str],
                                   window_functions: List[Dict[str, str]],
                                   where_clause: Optional[str] = None,
                                   order_by: Optional[str] = None) -> str:
        """Build query with window functions."""
        
        # Base columns
        select_columns = [self._quote_identifier(col) for col in columns]
        
        # Window functions
        for wf in window_functions:
            func = wf['function']  # ROW_NUMBER, RANK, LAG, LEAD, etc.
            partition_by = wf.get('partition_by')
            order_by_window = wf.get('order_by')
            alias = wf.get('alias', func.lower())
            
            window_expr = f"{func}()"
            
            # OVER clause
            over_parts = []
            if partition_by:
                partition_cols = ', '.join([self._quote_identifier(col) for col in partition_by])
                over_parts.append(f"PARTITION BY {partition_cols}")
            
            if order_by_window:
                over_parts.append(f"ORDER BY {order_by_window}")
            
            over_clause = ' '.join(over_parts)
            window_expr += f" OVER ({over_clause})"
            
            if alias:
                window_expr += f" AS {self._quote_identifier(alias)}"
            
            select_columns.append(window_expr)
        
        columns_str = ', '.join(select_columns)
        query = f"SELECT {columns_str} FROM {self._quote_identifier(table)}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        return query
    
    def build_cte_query(self,
                       ctes: List[Dict[str, str]],
                       main_query: str) -> str:
        """Build query with Common Table Expressions (CTEs)."""
        
        cte_definitions = []
        for cte in ctes:
            name = cte['name']
            query = cte['query']
            recursive = cte.get('recursive', False)
            
            cte_def = f"{self._quote_identifier(name)} AS ({query})"
            cte_definitions.append(cte_def)
        
        # Check if any CTE is recursive
        with_clause = "WITH RECURSIVE" if any(cte.get('recursive') for cte in ctes) else "WITH"
        
        ctes_str = ',\n'.join(cte_definitions)
        query = f"{with_clause}\n{ctes_str}\n{main_query}"
        
        return query
    
    def _quote_identifier(self, identifier: str) -> str:
        """Quote database identifier (table/column name)."""
        # Handle qualified names (schema.table or table.column)
        if '.' in identifier:
            parts = identifier.split('.')
            return '.'.join([f"{self.quote_char}{part}{self.quote_char}" for part in parts])
        else:
            return f"{self.quote_char}{identifier}{self.quote_char}"
    
    def escape_string_literal(self, value: str) -> str:
        """Escape string literal for SQL."""
        # Basic SQL string escaping (replace single quotes)
        return value.replace("'", "''")
    
    def format_value(self, value: Any) -> str:
        """Format value for SQL query."""
        if value is None:
            return 'NULL'
        elif isinstance(value, str):
            return f"'{self.escape_string_literal(value)}'"
        elif isinstance(value, bool):
            return 'TRUE' if value else 'FALSE'
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            # For other types, convert to string and quote
            return f"'{self.escape_string_literal(str(value))}'"