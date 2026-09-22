"""
Selenium End-to-End Test Suite for Administrator Portal (System Analytics & Model Performance).
Tests:
- Admin login flow (/login/admin with admin/admin123)
- System metrics KPI stat cards
- Registered User Accounts Management table
- Prediction Activity & Audit Trail log table
"""

import os
import sys
import time
import unittest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .conftest import get_chrome_driver, BASE_URL


class TestAdminPortalE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.headless = os.environ.get("HEADED_SELENIUM") != "1"
        cls.driver = get_chrome_driver(headless=cls.headless)
        cls.wait = WebDriverWait(cls.driver, 12)

    @classmethod
    def tearDownClass(cls):
        if cls.driver:
            cls.driver.quit()

    def setUp(self):
        self.driver.delete_all_cookies()

    def _login_as_admin(self):
        driver = self.driver
        wait = self.wait
        driver.get(f"{BASE_URL}/login/admin")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "h2")))
        user_input = wait.until(EC.visibility_of_element_located((By.ID, "username")))
        pass_input = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        user_input.clear()
        user_input.send_keys("admin")
        pass_input.clear()
        pass_input.send_keys("admin123")
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        wait.until(EC.url_contains("/admin"))

    def test_01_admin_login_flow(self):
        """Test logging into Admin Portal."""
        driver = self.driver
        wait = self.wait
        
        self._login_as_admin()
        
        h1_elem = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertIn("System Analytics", h1_elem.text)
        print("[PASSED] Admin Login & Dashboard Navigation Verified.")

    def test_02_kpi_stat_cards(self):
        """Test KPI statistic cards on Admin Dashboard."""
        driver = self.driver
        wait = self.wait
        
        self._login_as_admin()
        
        stat_cards = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "stat-card")))
        self.assertGreaterEqual(len(stat_cards), 4)
        print(f"[PASSED] Verified {len(stat_cards)} KPI Stat Cards on Admin Dashboard.")

    def test_03_registered_users_table(self):
        """Test user accounts table on Admin Dashboard."""
        driver = self.driver
        wait = self.wait
        
        self._login_as_admin()
        
        users_table = wait.until(EC.presence_of_element_located((By.ID, "usersTable")))
        rows = users_table.find_elements(By.CSS_SELECTOR, "tbody tr")
        self.assertGreater(len(rows), 0)
        
        page_source = driver.page_source
        self.assertIn("admin", page_source)
        self.assertIn("writer", page_source)
        self.assertIn("user", page_source)
        print(f"[PASSED] Registered Accounts Table Verified ({len(rows)} accounts found).")

    def test_04_prediction_audit_log_table(self):
        """Test activity audit trail table on Admin Dashboard."""
        driver = self.driver
        wait = self.wait
        
        self._login_as_admin()
        
        pred_table = wait.until(EC.presence_of_element_located((By.ID, "predictionsTable")))
        self.assertIsNotNone(pred_table)
        print("[PASSED] Prediction Activity & Audit Trail Table Verified.")


if __name__ == "__main__":
    unittest.main()
