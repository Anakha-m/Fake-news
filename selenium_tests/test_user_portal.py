"""
Selenium End-to-End Test Suite for User Portal (SDG 16 Fake News Detector).
Tests:
- User login flow
- Custom news submission & AJAX prediction rendering
- Real / Fake probability distribution & confidence meters
- Quick preset buttons ('Real ISRO News', 'Viral Garlic Hoax')
- Live RSS feed integration and auto-population
"""

import os
import sys
import time
import unittest
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .conftest import get_chrome_driver, BASE_URL


class TestUserPortalE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.headless = os.environ.get("HEADED_SELENIUM") != "1"
        cls.driver = get_chrome_driver(headless=cls.headless)
        cls.wait = WebDriverWait(cls.driver, 12)

    @classmethod
    def tearDownClass(cls):
        if cls.driver:
            cls.driver.quit()

    def test_01_user_login_flow(self):
        """Test logging into the User Portal with demo credentials."""
        driver = self.driver
        wait = self.wait
        
        driver.get(f"{BASE_URL}/login/user")
        
        # Verify page header
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "h2")))
        self.assertIn("User Portal", driver.page_source)
        
        # Enter credentials
        user_input = wait.until(EC.visibility_of_element_located((By.ID, "username")))
        pass_input = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        
        user_input.clear()
        user_input.send_keys("user")
        pass_input.clear()
        pass_input.send_keys("user123")
        
        # Submit
        submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        submit_btn.click()
        
        # Verify redirect to /user or /
        wait.until(lambda d: "/user" in d.current_url or d.current_url.endswith(":5000/"))
        self.assertIn("AI Fake News Analyzer", driver.page_source)
        print("[PASSED] User Login & Redirection Successful.")

    def test_02_submit_real_news_prediction(self):
        """Test analyzing authentic space news and verifying REAL verdict."""
        driver = self.driver
        wait = self.wait
        
        driver.get(f"{BASE_URL}/")
        
        # Fill news form
        title_input = wait.until(EC.visibility_of_element_located((By.ID, "newsTitle")))
        text_input = wait.until(EC.visibility_of_element_located((By.ID, "newsText")))
        
        title_input.clear()
        title_input.send_keys("ISRO Launches Advanced Earth Observation Satellite")
        
        text_input.clear()
        text_input.send_keys("The Indian Space Research Organisation successfully launched the new navigation satellite into geostationary transfer orbit from Sriharikota.")
        
        # Click Analyze News button
        analyze_btn = driver.find_element(By.ID, "analyzeBtn")
        analyze_btn.click()
        
        # Wait for Result Box to become visible
        result_box = wait.until(EC.visibility_of_element_located((By.ID, "resultBox")))
        self.assertTrue(result_box.is_displayed())
        
        # Verify prediction verdict
        verdict_elem = wait.until(EC.visibility_of_element_located((By.ID, "verdictTitle")))
        verdict_text = verdict_elem.text
        self.assertIn("REAL", verdict_text.upper())
        
        # Verify confidence percent & probability
        conf_elem = driver.find_element(By.ID, "confidencePercent")
        prob_real = driver.find_element(By.ID, "probReal")
        self.assertIn("%", conf_elem.text)
        self.assertIn("%", prob_real.text)
        print(f"[PASSED] Real News Analysis: Verdict='{verdict_text}', Conf={conf_elem.text}, RealProb={prob_real.text}")

    def test_03_submit_fake_news_and_check_review_notice(self):
        """Test analyzing a hoax claim and verifying FAKE verdict & editorial review notice."""
        driver = self.driver
        wait = self.wait
        
        driver.get(f"{BASE_URL}/")
        
        title_input = wait.until(EC.visibility_of_element_located((By.ID, "newsTitle")))
        text_input = wait.until(EC.visibility_of_element_located((By.ID, "newsText")))
        
        title_input.clear()
        title_input.send_keys("Miracle cure secret suppressed by hospitals!")
        
        text_input.clear()
        text_input.send_keys("SHOCKING: Eating raw garlic and lemon cures all incurable diseases overnight and doctors are hiding this miracle cure from everyone!")
        
        analyze_btn = driver.find_element(By.ID, "analyzeBtn")
        analyze_btn.click()
        
        # Wait for Result Box
        result_box = wait.until(EC.visibility_of_element_located((By.ID, "resultBox")))
        self.assertTrue(result_box.is_displayed())
        
        verdict_elem = wait.until(EC.visibility_of_element_located((By.ID, "verdictTitle")))
        verdict_text = verdict_elem.text
        self.assertIn("FAKE", verdict_text.upper())
        
        # Check that Content Writer review notice is visible
        review_notice = driver.find_element(By.ID, "humanReviewNotice")
        self.assertTrue(review_notice.is_displayed())
        self.assertIn("Content Writer", review_notice.text)
        print(f"[PASSED] Fake News Analysis: Verdict='{verdict_text}', Review Notice Dispatched.")

    def test_04_quick_sample_preset_buttons(self):
        """Test clicking sample presets auto-populates the input form."""
        driver = self.driver
        wait = self.wait
        
        driver.get(f"{BASE_URL}/")
        
        # Find preset buttons
        presets = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "preset-chip")))
        self.assertGreaterEqual(len(presets), 2)
        
        # Click first preset (ISRO)
        presets[0].click()
        time.sleep(0.5)
        
        text_input = driver.find_element(By.ID, "newsText")
        self.assertGreater(len(text_input.get_attribute("value")), 10)
        print("[PASSED] Preset Auto-population Verified.")

    def test_05_live_rss_feed_interaction(self):
        """Test live RSS feed section and 'Test' button auto-population."""
        driver = self.driver
        wait = self.wait
        
        driver.get(f"{BASE_URL}/")
        
        # Wait for RSS feed cards to be rendered dynamically
        try:
            cards = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#newsGrid .news-card")))
        except Exception:
            news_grid = wait.until(EC.presence_of_element_located((By.ID, "newsGrid")))
            cards = news_grid.find_elements(By.CLASS_NAME, "news-card")
        
        self.assertGreater(len(cards), 0, "Expected at least 1 RSS news card to be loaded")
        
        # Click the test button on the first card
        test_btn = cards[0].find_element(By.CSS_SELECTOR, "button")
        test_btn.click()
        time.sleep(0.5)
        
        # Form should be populated
        text_input = driver.find_element(By.ID, "newsText")
        self.assertGreater(len(text_input.get_attribute("value")), 5)
        print(f"[PASSED] Live RSS Feed Displayed ({len(cards)} items) and Test Trigger Verified.")


if __name__ == "__main__":
    unittest.main()
