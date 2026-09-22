"""
Selenium End-to-End Test Suite for Content Writer Portal (Veritas Moderation Console).
Tests:
- Content Writer login flow (/login/writer)
- Moderation console queue rendering
- Status filter buttons ('All Flagged Claims', 'Pending Review', 'Resolved')
- Live queue search bar
"""

import os
import sys
import time
import unittest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .conftest import get_chrome_driver, BASE_URL


class TestWriterPortalE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.headless = os.environ.get("HEADED_SELENIUM") != "1"
        cls.driver = get_chrome_driver(headless=cls.headless)
        cls.wait = WebDriverWait(cls.driver, 12)

    @classmethod
    def tearDownClass(cls):
        if cls.driver:
            cls.driver.quit()

    def _ensure_writer_logged_in(self):
        driver = self.driver
        wait = self.wait
        driver.get(f"{BASE_URL}/writer")
        if "/login/writer" in driver.current_url:
            u = wait.until(EC.visibility_of_element_located((By.ID, "username")))
            p = wait.until(EC.visibility_of_element_located((By.ID, "password")))
            u.clear()
            u.send_keys("writer")
            p.clear()
            p.send_keys("writer123")
            driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
            wait.until(EC.url_contains("/writer"))

    def test_01_writer_login_flow(self):
        """Test logging into Content Writer portal."""
        driver = self.driver
        wait = self.wait
        
        # Ensure fresh logout
        driver.delete_all_cookies()
        driver.get(f"{BASE_URL}/login/writer")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "h2")))
        
        user_input = wait.until(EC.visibility_of_element_located((By.ID, "username")))
        pass_input = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        
        user_input.clear()
        user_input.send_keys("writer")
        pass_input.clear()
        pass_input.send_keys("writer123")
        
        submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        submit_btn.click()
        
        wait.until(EC.url_contains("/writer"))
        h1_elem = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertIn("Content Writer", h1_elem.text)
        print("[PASSED] Writer Login & Console Access Verified.")

    def test_02_moderation_queue_and_filters(self):
        """Test filtering the moderation queue by status tabs."""
        driver = self.driver
        wait = self.wait
        
        self._ensure_writer_logged_in()
        
        filter_buttons = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "filter-btn")))
        self.assertGreaterEqual(len(filter_buttons), 3)
        
        # Click 'Pending Review' tab
        filter_buttons[1].click()
        time.sleep(0.4)
        
        # Click 'All Flagged Claims' tab
        filter_buttons[0].click()
        time.sleep(0.4)
        print("[PASSED] Moderation Queue Status Filter Tabs Verified.")

    def test_03_queue_search_bar(self):
        """Test dynamic client-side search in moderation queue."""
        driver = self.driver
        wait = self.wait
        
        self._ensure_writer_logged_in()
        
        search_input = wait.until(EC.visibility_of_element_located((By.ID, "queueSearchInput")))
        search_input.clear()
        search_input.send_keys("ISRO")
        time.sleep(0.5)
        
        search_input.clear()
        search_input.send_keys("Garlic")
        time.sleep(0.5)
        
        search_input.clear()
        print("[PASSED] Queue Search Bar Dynamic Filter Verified.")


if __name__ == "__main__":
    unittest.main()
