"""
Unit tests for query builder functionality.
"""

import pytest
from src.qudata.database.query_builder import QueryBuilder


class TestQueryBuilder:
    """Test query builder functionality."""
    
    def test_init_postgresql(self):
        """Test query builder initialization for PostgreSQL."""
        builder = QueryBuilder('postgresql')
        assert builder.database_type == 'postgresql'
        assert builder.quote_char == '"'
    
    def test_init_mysql(self):
        """Test query builder initialization for MySQL."""
        builder = QueryBuilder('mysql')
        assert builder.database_type == 'mysql'
        assert builder.quote_char == '`'
    
    def test_init_invalid_database_type(self):
        """Test query builder initialization with invalid database type."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            QueryBuilder('invalid_db')
    
    def test_build_select_query_basic(self):
        """Test basic SELECT query building."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(
            table='users',
            columns=['id', 'name', 'email']
        )
        
        expected = 'SELECT "id", "name", "email" FROM "users"'
        assert query == expected
    
    def test_build_select_query_all_columns(self):
        """Test SELECT query with all columns."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(table='users')
        
        expected = 'SELECT * FROM "users"'
        assert query == expected
    
    def test_build_select_query_with_where(self):
        """Test SELECT query with WHERE clause."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(
            table='users',
            columns=['name', 'age'],
            where_clause='age > 25 AND active = true'
        )
        
        expected = 'SELECT "name", "age" FROM "users" WHERE age > 25 AND active = true'
        assert query == expected
    
    def test_build_select_query_with_order_by(self):
        """Test SELECT query with ORDER BY clause."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(
            table='users',
            columns=['name', 'age'],
            order_by='age DESC, name ASC'
        )
        
        expected = 'SELECT "name", "age" FROM "users" ORDER BY age DESC, name ASC'
        assert query == expected
    
    def test_build_select_query_with_group_by(self):
        """Test SELECT query with GROUP BY clause."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(
            table='orders',
            columns=['customer_id', 'COUNT(*)'],
            group_by=['customer_id']
        )
        
        expected = 'SELECT "customer_id", "COUNT(*)" FROM "orders" GROUP BY "customer_id"'
        assert query == expected
    
    def test_build_select_query_with_having(self):
        """Test SELECT query with HAVING clause."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(
            table='orders',
            columns=['customer_id', 'COUNT(*)'],
            group_by=['customer_id'],
            having='COUNT(*) > 5'
        )
        
        expected = 'SELECT "customer_id", "COUNT(*)" FROM "orders" GROUP BY "customer_id" HAVING COUNT(*) > 5'
        assert query == expected
    
    def test_build_select_query_with_limit_postgresql(self):
        """Test SELECT query with LIMIT for PostgreSQL."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(
            table='users',
            limit=10,
            offset=20
        )
        
        expected = 'SELECT * FROM "users" LIMIT 10 OFFSET 20'
        assert query == expected
    
    def test_build_select_query_with_limit_mysql(self):
        """Test SELECT query with LIMIT for MySQL."""
        builder = QueryBuilder('mysql')
        
        query = builder.build_select_query(
            table='users',
            limit=10,
            offset=20
        )
        
        expected = 'SELECT * FROM `users` LIMIT 20, 10'
        assert query == expected
    
    def test_build_select_query_with_limit_no_offset_mysql(self):
        """Test SELECT query with LIMIT but no OFFSET for MySQL."""
        builder = QueryBuilder('mysql')
        
        query = builder.build_select_query(
            table='users',
            limit=10
        )
        
        expected = 'SELECT * FROM `users` LIMIT 10'
        assert query == expected
    
    def test_build_select_query_distinct(self):
        """Test SELECT DISTINCT query."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_select_query(
            table='users',
            columns=['department'],
            distinct=True
        )
        
        expected = 'SELECT DISTINCT "department" FROM "users"'
        assert query == expected
    
    def test_build_count_query_basic(self):
        """Test basic COUNT query building."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_count_query(table='users')
        
        expected = 'SELECT COUNT(*) FROM "users"'
        assert query == expected
    
    def test_build_count_query_with_where(self):
        """Test COUNT query with WHERE clause."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_count_query(
            table='users',
            where_clause='active = true'
        )
        
        expected = 'SELECT COUNT(*) FROM "users" WHERE active = true'
        assert query == expected
    
    def test_build_count_query_distinct(self):
        """Test COUNT DISTINCT query."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_count_query(
            table='orders',
            distinct_column='customer_id'
        )
        
        expected = 'SELECT COUNT(DISTINCT "customer_id") FROM "orders"'
        assert query == expected
    
    def test_build_join_query_inner_join(self):
        """Test query building with INNER JOIN."""
        builder = QueryBuilder('postgresql')
        
        join_specs = [
            {
                'table': 'orders',
                'join_type': 'INNER',
                'on_condition': 'users.id = orders.user_id'
            }
        ]
        
        query = builder.build_join_query(
            main_table='users',
            join_specs=join_specs,
            columns=['users.name', 'orders.total']
        )
        
        expected = 'SELECT "users.name", "orders.total" FROM "users" INNER JOIN "orders" ON users.id = orders.user_id'
        assert query == expected
    
    def test_build_join_query_multiple_joins(self):
        """Test query building with multiple JOINs."""
        builder = QueryBuilder('postgresql')
        
        join_specs = [
            {
                'table': 'orders',
                'join_type': 'LEFT',
                'on_condition': 'users.id = orders.user_id'
            },
            {
                'table': 'products',
                'join_type': 'INNER',
                'on_condition': 'orders.product_id = products.id',
                'alias': 'p'
            }
        ]
        
        query = builder.build_join_query(
            main_table='users',
            join_specs=join_specs,
            columns=['users.name', 'orders.total', 'p.name']
        )
        
        expected = ('SELECT "users.name", "orders.total", "p.name" FROM "users" '
                   'LEFT JOIN "orders" ON users.id = orders.user_id '
                   'INNER JOIN "products" AS "p" ON orders.product_id = products.id')
        assert query == expected
    
    def test_build_insert_query_basic(self):
        """Test basic INSERT query building."""
        builder = QueryBuilder('postgresql')
        
        data = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 30
        }
        
        query = builder.build_insert_query(table='users', data=data)
        
        expected = 'INSERT INTO "users" ("name", "email", "age") VALUES (%s, %s, %s)'
        assert query == expected
    
    def test_build_insert_query_with_conflict_postgresql(self):
        """Test INSERT query with conflict handling for PostgreSQL."""
        builder = QueryBuilder('postgresql')
        
        data = {'name': 'John', 'email': 'john@example.com'}
        
        query = builder.build_insert_query(
            table='users',
            data=data,
            on_conflict='(email) DO UPDATE SET name = EXCLUDED.name'
        )
        
        expected = ('INSERT INTO "users" ("name", "email") VALUES (%s, %s) '
                   'ON CONFLICT (email) DO UPDATE SET name = EXCLUDED.name')
        assert query == expected
    
    def test_build_insert_query_with_conflict_mysql(self):
        """Test INSERT query with conflict handling for MySQL."""
        builder = QueryBuilder('mysql')
        
        data = {'name': 'John', 'email': 'john@example.com'}
        
        query = builder.build_insert_query(
            table='users',
            data=data,
            on_conflict='name = VALUES(name)'
        )
        
        expected = ('INSERT INTO `users` (`name`, `email`) VALUES (%s, %s) '
                   'ON DUPLICATE KEY UPDATE name = VALUES(name)')
        assert query == expected
    
    def test_build_update_query(self):
        """Test UPDATE query building."""
        builder = QueryBuilder('postgresql')
        
        data = {
            'name': 'Jane Doe',
            'email': 'jane@example.com'
        }
        
        query = builder.build_update_query(
            table='users',
            data=data,
            where_clause='id = 1'
        )
        
        expected = 'UPDATE "users" SET "name" = %s, "email" = %s WHERE id = 1'
        assert query == expected
    
    def test_build_delete_query(self):
        """Test DELETE query building."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_delete_query(
            table='users',
            where_clause='active = false AND last_login < \'2023-01-01\''
        )
        
        expected = 'DELETE FROM "users" WHERE active = false AND last_login < \'2023-01-01\''
        assert query == expected
    
    def test_build_create_table_query_basic(self):
        """Test basic CREATE TABLE query building."""
        builder = QueryBuilder('postgresql')
        
        columns = [
            {'name': 'id', 'type': 'SERIAL', 'not_null': True},
            {'name': 'name', 'type': 'VARCHAR(100)', 'not_null': True},
            {'name': 'email', 'type': 'VARCHAR(255)', 'unique': True},
            {'name': 'created_at', 'type': 'TIMESTAMP', 'default': 'CURRENT_TIMESTAMP'}
        ]
        
        query = builder.build_create_table_query(
            table='users',
            columns=columns,
            primary_key=['id']
        )
        
        expected_lines = [
            'CREATE TABLE "users" (',
            '    "id" SERIAL NOT NULL,',
            '    "name" VARCHAR(100) NOT NULL,',
            '    "email" VARCHAR(255) UNIQUE,',
            '    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,',
            '    PRIMARY KEY ("id")',
            ')'
        ]
        expected = '\n'.join(expected_lines)
        assert query == expected
    
    def test_build_create_table_query_with_foreign_keys(self):
        """Test CREATE TABLE query with foreign keys."""
        builder = QueryBuilder('postgresql')
        
        columns = [
            {'name': 'id', 'type': 'SERIAL'},
            {'name': 'user_id', 'type': 'INTEGER'},
            {'name': 'title', 'type': 'VARCHAR(200)'}
        ]
        
        foreign_keys = [
            {
                'column': 'user_id',
                'ref_table': 'users',
                'ref_column': 'id',
                'on_delete': 'CASCADE',
                'on_update': 'RESTRICT'
            }
        ]
        
        query = builder.build_create_table_query(
            table='posts',
            columns=columns,
            primary_key=['id'],
            foreign_keys=foreign_keys
        )
        
        assert 'FOREIGN KEY ("user_id") REFERENCES "users" ("id")' in query
        assert 'ON DELETE CASCADE' in query
        assert 'ON UPDATE RESTRICT' in query
    
    def test_build_create_index_query_basic(self):
        """Test basic CREATE INDEX query building."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_create_index_query(
            index_name='idx_users_email',
            table='users',
            columns=['email']
        )
        
        expected = 'CREATE INDEX "idx_users_email" ON "users" ("email")'
        assert query == expected
    
    def test_build_create_index_query_unique(self):
        """Test CREATE UNIQUE INDEX query building."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_create_index_query(
            index_name='idx_users_email_unique',
            table='users',
            columns=['email'],
            unique=True
        )
        
        expected = 'CREATE UNIQUE INDEX "idx_users_email_unique" ON "users" ("email")'
        assert query == expected
    
    def test_build_create_index_query_with_where(self):
        """Test CREATE INDEX query with WHERE clause."""
        builder = QueryBuilder('postgresql')
        
        query = builder.build_create_index_query(
            index_name='idx_users_active',
            table='users',
            columns=['created_at'],
            where_clause='active = true'
        )
        
        expected = 'CREATE INDEX "idx_users_active" ON "users" ("created_at") WHERE active = true'
        assert query == expected
    
    def test_build_aggregation_query(self):
        """Test aggregation query building."""
        builder = QueryBuilder('postgresql')
        
        aggregations = [
            {'function': 'COUNT', 'column': '*', 'alias': 'total_orders'},
            {'function': 'SUM', 'column': 'amount', 'alias': 'total_amount'},
            {'function': 'AVG', 'column': 'amount', 'alias': 'avg_amount'}
        ]
        
        query = builder.build_aggregation_query(
            table='orders',
            aggregations=aggregations,
            group_by=['customer_id'],
            having='COUNT(*) > 1'
        )
        
        expected = ('SELECT "customer_id", COUNT(*) AS "total_orders", '
                   'SUM("amount") AS "total_amount", AVG("amount") AS "avg_amount" '
                   'FROM "orders" GROUP BY "customer_id" HAVING COUNT(*) > 1')
        assert query == expected
    
    def test_build_window_function_query(self):
        """Test window function query building."""
        builder = QueryBuilder('postgresql')
        
        window_functions = [
            {
                'function': 'ROW_NUMBER',
                'partition_by': ['department'],
                'order_by': 'salary DESC',
                'alias': 'rank_in_dept'
            },
            {
                'function': 'LAG',
                'partition_by': ['department'],
                'order_by': 'hire_date',
                'alias': 'prev_hire'
            }
        ]
        
        query = builder.build_window_function_query(
            table='employees',
            columns=['name', 'department', 'salary'],
            window_functions=window_functions
        )
        
        expected_parts = [
            'SELECT "name", "department", "salary"',
            'ROW_NUMBER() OVER (PARTITION BY "department" ORDER BY salary DESC) AS "rank_in_dept"',
            'LAG() OVER (PARTITION BY "department" ORDER BY hire_date) AS "prev_hire"',
            'FROM "employees"'
        ]
        
        for part in expected_parts:
            assert part in query
    
    def test_build_cte_query(self):
        """Test Common Table Expression (CTE) query building."""
        builder = QueryBuilder('postgresql')
        
        ctes = [
            {
                'name': 'high_value_customers',
                'query': 'SELECT customer_id FROM orders GROUP BY customer_id HAVING SUM(amount) > 1000'
            },
            {
                'name': 'recent_orders',
                'query': 'SELECT * FROM orders WHERE created_at > \'2023-01-01\''
            }
        ]
        
        main_query = 'SELECT * FROM high_value_customers hvc JOIN recent_orders ro ON hvc.customer_id = ro.customer_id'
        
        query = builder.build_cte_query(ctes, main_query)
        
        expected_parts = [
            'WITH',
            '"high_value_customers" AS (',
            '"recent_orders" AS (',
            main_query
        ]
        
        for part in expected_parts:
            assert part in query
    
    def test_build_cte_query_recursive(self):
        """Test recursive CTE query building."""
        builder = QueryBuilder('postgresql')
        
        ctes = [
            {
                'name': 'employee_hierarchy',
                'query': 'SELECT id, name, manager_id FROM employees WHERE manager_id IS NULL',
                'recursive': True
            }
        ]
        
        main_query = 'SELECT * FROM employee_hierarchy'
        
        query = builder.build_cte_query(ctes, main_query)
        
        assert query.startswith('WITH RECURSIVE')
    
    def test_quote_identifier_simple(self):
        """Test simple identifier quoting."""
        builder = QueryBuilder('postgresql')
        
        quoted = builder._quote_identifier('table_name')
        assert quoted == '"table_name"'
    
    def test_quote_identifier_qualified(self):
        """Test qualified identifier quoting."""
        builder = QueryBuilder('postgresql')
        
        quoted = builder._quote_identifier('schema.table')
        assert quoted == '"schema"."table"'
        
        quoted = builder._quote_identifier('table.column')
        assert quoted == '"table"."column"'
    
    def test_escape_string_literal(self):
        """Test string literal escaping."""
        builder = QueryBuilder('postgresql')
        
        escaped = builder.escape_string_literal("O'Reilly")
        assert escaped == "O''Reilly"
        
        escaped = builder.escape_string_literal("It's a test")
        assert escaped == "It''s a test"
    
    def test_format_value_various_types(self):
        """Test value formatting for different data types."""
        builder = QueryBuilder('postgresql')
        
        # String
        assert builder.format_value('hello') == "'hello'"
        assert builder.format_value("O'Reilly") == "'O''Reilly'"
        
        # Numbers
        assert builder.format_value(42) == '42'
        assert builder.format_value(3.14) == '3.14'
        
        # Boolean
        assert builder.format_value(True) == 'TRUE'
        assert builder.format_value(False) == 'FALSE'
        
        # None
        assert builder.format_value(None) == 'NULL'
        
        # Other types (converted to string)
        from datetime import date
        test_date = date(2023, 1, 1)
        assert builder.format_value(test_date) == "'2023-01-01'"