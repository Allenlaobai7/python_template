import unittest
import pandas as pd
import data_conn

class TestDataConn(unittest.TestCase):
    def test_datetimeoffset(self):
        """
        Test pulling datetimeoffset data
        """
        conn = data_conn.get_connection('prd')
        data = pd.read_sql("select top 10 * from database1.table1", conn)
        self.assertEqual(len(data), 10)

    def test_run_query_file(self):
        """
        Test running query file to pull data
        """
        conn = data_conn.get_connection('prd')
        data = data_conn.get_data_by_query_file(conn, 'test.sql', "2022")
        self.assertEqual(len(data), 10)

if __name__ == '__main__':
    unittest.main()
