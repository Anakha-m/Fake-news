"""
Smoke Test for Veritas Fake News Detector Login Portals using Selenium WebDriver.
Verifies the login portal navigation, page title, and tab elements on port 5000.
"""

import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = os.environ.get("TEST_SERVER_URL", "http://127.0.0.1:5000")


def run_smoke_test(headless=True):
    print(f"\n[INFO] Starting Selenium Smoke Test against {BASE_URL}...")
    
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,800")
    
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 10)
    
    try:
        # 1. Open User Login Portal
        driver.get(f"{BASE_URL}/login/user")
        print(f"[PASSED] Navigated to {BASE_URL}/login/user")
        
        # 2. Verify Page Title
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        print(f"[PASSED] Page Title: '{driver.title}'")
        assert "Veritas" in driver.title or "Login" in driver.title, f"Unexpected title: {driver.title}"
        
        # 3. Verify Form Elements
        username_field = wait.until(EC.visibility_of_element_located((By.ID, "username")))
        password_field = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        
        assert username_field is not None
        assert password_field is not None
        assert submit_button is not None
        print("[PASSED] Login form fields (username, password, submit) verified.")
        
        # 4. Check Portal Switcher Links
        portal_links = driver.find_elements(By.CSS_SELECTOR, "div a[href*='/login/']")
        assert len(portal_links) >= 3, f"Expected at least 3 portal tabs, found {len(portal_links)}"
        print(f"[PASSED] Verified {len(portal_links)} portal navigation tabs (User, Writer, Admin).")
        
        print("\n==========================================")
        print("ALL SMOKE TESTS PASSED SUCCESSFULLY (100%)")
        print("==========================================\n")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Smoke test failed: {e}")
        return False
        
    finally:
        driver.quit()


if __name__ == "__main__":
    is_headless = "--headed" not in sys.argv
    success = run_smoke_test(headless=is_headless)
    sys.exit(0 if success else 1)