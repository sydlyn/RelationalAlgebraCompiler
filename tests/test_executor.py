# tests/test_executor.py

import unittest
from tests.test_base import BaseTest
import ra_compiler.cli as cli
import ra_compiler.executor as exe
# import ra_compiler.exceptions as exceptions

class TestExecutor(BaseTest):

    def test_execute_empty(self):
        self.assertRaises(ValueError, exe.execute, None)
        self.assertRaises(ValueError, exe.execute, {})
        self.assertRaises(ValueError, exe.execute, "")

    def test_prepare_cols_for_merge(self):
        cli.handle_query("(T1 /x T2)->T3")

        ndf1 = exe.load_table("T3")
        ndf2 = exe.load_table("T1")

        exe.prepare_for_merge_op(ndf1.df, ndf2.df)

        expected_left_cols = ["A_L_L", "B_L", "A_R_L", "C"]
        expected_right_cols = ["A_R", "B_R"]

        for col in expected_left_cols:
            self.assertIn(col, ndf1.df.columns,
                          f"Expected '{col}' in columns {list( ndf1.df.columns)}")

        for col in expected_right_cols:
            self.assertIn(col, ndf2.df.columns,
                          f"Expected '{col}' in columns {list( ndf2.df.columns)}")

if __name__ == "__main__":
    unittest.main()
