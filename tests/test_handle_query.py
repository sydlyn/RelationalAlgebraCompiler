# tests/test_handle_query.py

import unittest
import pandas as pd
import ra_compiler.cli as cli
import ra_compiler.mysql as db

class TestHandleQuery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db.setup_mysql()
        
        cls.conn = getattr(db, "conn", getattr(db, "connection", None))
        if cls.conn is None:
            raise unittest.SkipTest("No DB connection available")

        cur = cls.conn.cursor()

        # Prepare testTable
        cur.execute("DROP TABLE IF EXISTS testTable;")
        cur.execute("CREATE TABLE testTable (a INT, b INT);")
        cur.executemany("INSERT INTO testTable (a, b) VALUES (%s, %s);", 
                        [(0,1,), (1,2,), (2,2,), (4,3,)])

        # Prepare R table
        cur.execute("DROP TABLE IF EXISTS R;")
        cur.execute("""CREATE TABLE R 
                    (age INT, b INT, c INT, d INT, name VARCHAR(50));""")
        rac_rows = [
            (12, 5, 19, 4, "alice"),
            (7, 15, 2, 17, "bob"),
            (11, 20, 8, 6, "claire"),
            (3, 18, 9, 10, "dave"),
            (14, 7, 1, 13, "emma"),
            (8, 16, 5, 12, "faith"),
            (19, 6, 17, 2, "george"),
            (4, 10, 11, 3, "hannah"),
            (1, 13, 14, 9, "ian"),
            (1, 13, 14, 9, "jack"),
            (6, 3, 20, 15, "kelly"),
            (6, 3, 20, 15, "kelly"),
            (8, 3, 9, 4, "bob"),
            (8, 4, 9, 4, "bob"),
            (8, 5, 9, 4, "bob"),
            (12, None, 3, None, None),
        ]
        cur.executemany("INSERT INTO R VALUES (%s, %s, %s, %s, %s)", rac_rows)

        # Prepare T table
        cur.execute("DROP TABLE IF EXISTS T;")
        cur.execute("""CREATE TABLE T (age INT, b INT, c INT);""")
        T_rows = [
            (1, 3, 100),
            (3, 4, 200),
            (6, 5, 300),
            (12, 6, 400),
        ]
        cur.executemany("INSERT INTO T VALUES (%s, %s, %s)", T_rows)

        cls.conn.commit()
        cur.close()

    @classmethod
    def tearDownClass(cls):
        # Clean up testTable
        if cls.conn:
            cur = cls.conn.cursor()
            cur.execute("DROP TABLE IF EXISTS testTable;")
            cur.execute("DROP TABLE IF EXISTS R;")
            cur.execute("DROP TABLE IF EXISTS T;")
            cls.conn.commit()
            cur.close()

        db.close_mysql()

    def _run_and_check(self, query, expected_cols):
        """Helper to run a query and verify columns exist."""

        result = cli.handle_query(query)

        self.assertIsNotNone(result, f"Query returned None: {query}")
        self.assertTrue(hasattr(result, "df"), f"Result missing .df: {query}")

        df = result.df

        for col in expected_cols:
            self.assertIn(col, df.columns, f"Expected '{col}' in columns {list(df.columns)}")

        print(df)
        return df

    def test_handle_query_projection(self):
        """Test handling a projection query."""

        df = self._run_and_check("(/pi {b} testTable)", ["b"])

        expected = pd.DataFrame({"b": [1, 2, 3]})
        pd.testing.assert_frame_equal(df, expected, check_dtype=False, check_index_type=False)

    def test_sigma_true(self):
        df = self._run_and_check("(/sigma{True} (R))", ["age", "b", "c", "d", "name"])
        self.assertEqual(len(df), 16)

    def test_sigma_false(self):
        df = self._run_and_check("(/sigma{2=4} (R))", ["age", "b", "c", "d", "name"])
        self.assertEqual(len(df), 0)

    def test_selection_age_gt_10(self):
        df = self._run_and_check("(/selection {age > 10} R)", ["age", "b", "c", "d", "name"])
        self.assertTrue((df["age"] > 10).all())

    def test_projection_after_selection(self):
        df = self._run_and_check("(/pi {name, age} (/selection {age > 10} R))", ["name", "age"])
        self.assertTrue((df["age"] > 10).all())

    def test_group_sum_avg(self):
        df = self._run_and_check("(/group {age, b; sum(c), avg(d)} R)", ["age", "b", "sum_c", "mean_d"])
        self.assertGreater(len(df), 0)

    def test_rho_and_join(self):
        cli.handle_query("(/rho {T1} R)")
        cli.handle_query("(/rho {T2} T)")
        df = self._run_and_check("(T1 /join {T1.c = T2.b} T2)", [])
        self.assertGreater(len(df), 0)

    def test_full_outer_join(self):
        df = self._run_and_check("(R /full_outer_join T)", ["age", "b", "c"])
        self.assertGreater(len(df), 0)

    def test_left_join(self):
        df = self._run_and_check("(R /left T)", ["age", "b", "c"])
        self.assertGreater(len(df), 0)

    def test_right_join_with_rho(self):
        cli.handle_query("(/rho {t2b} (/pi {b} T))")
        df = self._run_and_check("(R /right t2b)", ["age", "b"])
        self.assertGreater(len(df), 0)

    def test_left_join_reverse(self):
        cli.handle_query("(/rho {t2b2} (/pi {b} T))")
        df = self._run_and_check("(t2b2 /left R)", ["b"])
        self.assertGreater(len(df), 0)

    def test_delta_rho(self):
        df = self._run_and_check("(/rho {noDup} (/delta R))", ["age", "b", "c", "d", "name"])
        self.assertLessEqual(len(df), len(pd.read_sql("SELECT * FROM R", self.conn)))

    def test_division(self):
        df = self._run_and_check("(R /div (/selection {b = 3} (/pi^d {b} T)))", ["age", "c", "d", "name"])
        self.assertGreaterEqual(len(df), 0)

    def test_various_joins(self):
        join_queries = [
            "(R /join T)",
            "(R /join {R.age=T.age} T)",
            "(R /join {R.b = T.b} T)",
            "(R /join {R.c = T.b} T)",
            "(R /join {b} T)",
            "((/pi {b, c} R) /join {R.b = T.b} (/pi {b} T))",
            "((/pi {b} R) /join {R.b = T.b} (/pi {age, b} T))",
            "((/rho {t1} (/pi {b, c} R)) /join {t1.b = t2.b} (/rho {t2} (/pi {b} T)))"
        ]
        for q in join_queries:
            with self.subTest(query=q):
                df = self._run_and_check(q, [])
                self.assertGreaterEqual(len(df), 0)

    def test_intersect(self):
        df = self._run_and_check("((/pi{age} R) /intersect (/pi{age} R))", ["age"])
        self.assertGreater(len(df), 0)

    def test_semi_join(self):
        df = self._run_and_check("(T /semi {b} R)", ["age", "b", "c"])
        self.assertGreaterEqual(len(df), 0)

    def test_mass_projection(self):
        various_queries = [
            "(/pi{name} R)",
            "(/pi{name, age, b} R)",
            "(/projection{(2 + c)-> tot} R)",
            "(/projection{age + (2 + c)-> tot} R)",
            "(/projection{age + 2 + c-> tot} R)",
            "(/projection{(age + 2) + c-> tot} R)",
            "(/projection{(age + 2 + c)-> tot} R)",
            "(/projection{age + b -> tot} R)",
            "(/pi{age + b -> tot, name, d} R)",
            "(/projection{R.name, R.b} R) ",
            "(/pi{name, age} (/sigma{age > 3} R))",
            "(/pi{name} (/pi{name, age} R))",
            "(/projection{name, name} R)",
            "(/projection^d{d, c} R)",
            "(/pi{name, age + b -> total}(/sigma{(age + b) > 10}R))",
            "(/projection{name} (/selection{age > 2} (/projection{name, age, b} R)))",
        ]
        for q in various_queries:
            with self.subTest(query=q):
                self._run_and_check(q, [])

    def test_mass_selection(self):
        various_queries = [
            "(/selection{age > 3} R)",
            "(/sigma{age = 12} R)",
            "(/sigma{age > 3 and b < 16} R)",
            "(/selection{age < 8 or age > 15} R)",
            "(/sigma{(age > 3 and b < 16) or name = 'ian'} R)",
            "(/sigma{age > 2} (/sigma{b < 15} R))",
            "(/selection{age > b} R)",
            "(/sigma{R.age >= R.b} R)",
            "(/selection{age + b > 20} R)",
            "(/selection{20 < age + b} R)",
            "(/sigma{(age * 2) < 12} R)",
            "(/selection {age + b > b + c} R)",
            "(/selection {age + b > (b + c)} R)",
            "(/selection{(age + b) > c} R)",
        ]
        for q in various_queries:
            with self.subTest(query=q):
                self._run_and_check(q, [])

    if __name__ == "__main__":
        unittest.main()
