import unittest
import hpat.tests

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(hpat.tests))
    return suite

if __name__ == '__main__':
    unittest.main()
