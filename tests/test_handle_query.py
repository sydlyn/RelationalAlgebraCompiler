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
        cur.execute("DROP TABLE IF EXISTS testTable;")
        cur.execute("CREATE TABLE testTable (a INT, b INT);")
        cur.executemany("INSERT INTO testTable (a, b) VALUES (%s, %s);", [(0,1,), (1,2,), (2,2,), (4,3,)])
        cls.conn.commit()
        cur.close()

    @classmethod
    def tearDownClass(cls):
        # Clean up testTable
        if cls.conn:
            cur = cls.conn.cursor()
            cur.execute("DROP TABLE IF EXISTS testTable;")
            cls.conn.commit()
            cur.close()

        db.close_mysql()

    def test_handle_query_projection(self):
        """Test handling a projection query."""

        result = cli.handle_query("(/pi {b} testTable)")

        self.assertIsNotNone(result, "handle_query returned None")
        self.assertTrue(hasattr(result, "df"), "Result missing .df")

        df = result.df
        self.assertIn("b", df.columns, f"Expected 'b' column, got {list(df.columns)}")

        expected = pd.DataFrame({"b": [1, 2, 2, 3]})

        print(df)
        pd.testing.assert_frame_equal(df, expected, check_dtype=False)


    if __name__ == "__main__":
        unittest.main()
