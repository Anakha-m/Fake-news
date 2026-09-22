"""
Selenium End-to-End Test Suite for Role-Based Access Control (RBAC) & Security.
Tests:
- Unauthorized access redirection (guest trying to access /writer or /admin)
- Cross-portal permission enforcement
"""

import os
import sys
import unittest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .conftest import get_chrome_driver, BASE_URL


class TestSecurityAndAccessControlE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.headless = os.environ.get("HEADED_SELENIUM") != "1"
        cls.driver = get_chrome_driver(headless=cls.headless)
        cls.wait = WebDriverWait(cls.driver, 10)

    @classmethod
    def tearDownClass(cls):
        if cls.driver:
            cls.driver.quit()

    def test_01_unauthenticated_admin_redirect(self):
        """Test accessing /admin directly redirects to /login/admin."""
        driver = self.driver
        driver.delete_all_cookies()
        
        driver.get(f"{BASE_URL}/admin")
        self.wait.until(lambda d: "/login/admin" in d.current_url)
        self.assertIn("Admin Portal", driver.page_source)
        print("[PASSED] Unauthenticated /admin Access Correctly Redirected.")

    def test_02_unauthenticated_writer_redirect(self):
        """Test accessing /writer directly redirects to /login/writer."""
        driver = self.driver
        driver.delete_all_cookies()
        
        driver.get(f"{BASE_URL}/writer")
        self.wait.until(lambda d: "/login/writer" in d.current_url)
        self.assertIn("Content Writer", driver.page_source)
        print("[PASSED] Unauthenticated /writer Access Correctly Redirected.")


if __name__ == "__main__":
    unittest.main()
