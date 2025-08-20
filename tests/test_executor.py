# tests/test_executor.py

import unittest
import ra_compiler.cli as cli
import ra_compiler.mysql as db
import ra_compiler.executor as exe
# import ra_compiler.exceptions as exceptions

class TestExecutor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db.setup_mysql()

        cls.CONN = getattr(db, "CONN", getattr(db, "connection", None))
        if cls.CONN is None:
            raise unittest.SkipTest("No DB connection available")

        cur = cls.CONN.cursor()

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
        t_rows = [
            (1, 3, 100),
            (3, 4, 200),
            (6, 5, 300),
            (12, 6, 400),
        ]
        cur.executemany("INSERT INTO T VALUES (%s, %s, %s)", t_rows)

        cls.CONN.commit()
        cur.close()

    @classmethod
    def tearDownClass(cls):
        if cls.CONN:
            cur = cls.CONN.cursor()
            cur.execute("DROP TABLE IF EXISTS testTable;")
            cur.execute("DROP TABLE IF EXISTS R;")
            cur.execute("DROP TABLE IF EXISTS T;")
            cls.CONN.commit()
            cur.close()

        db.close_mysql()


    def test_execute_empty(self):
        self.assertRaises(ValueError, exe.execute, None)
        self.assertRaises(ValueError, exe.execute, {})
        self.assertRaises(ValueError, exe.execute, "")

    def test_prepare_cols_for_merge(self):
        cli.handle_query("(T1 /x T2)->T3")

        ndf1 = exe.load_table("T3")
        ndf2 = exe.load_table("T1")

        exe.prepare_cols_for_merge(ndf1.df, ndf2.df)

        expected_left_cols = ["A_L_L", "B_L", "A_R_L", "C"]
        expected_right_cols = ["A_R", "B_R"]

        for col in expected_left_cols:
            self.assertIn(col, ndf1.df.columns,
                          f"Expected '{col}' in columns {list( ndf1.df.columns)}")

        for col in expected_right_cols:
            self.assertIn(col, ndf2.df.columns,
                          f"Expected '{col}' in columns {list( ndf2.df.columns)}")
