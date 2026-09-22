"""
Selenium Test Automation Configuration & Fixtures.
Provides reusable browser drivers and server lifecycle helpers.
"""

import os
import sys
import time
import socket
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_PORT = 5000
BASE_URL = os.environ.get("TEST_SERVER_URL", f"http://127.0.0.1:{DEFAULT_PORT}")


def is_port_open(host="127.0.0.1", port=DEFAULT_PORT):
    """Check if the local server port is active."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect((host, port))
        s.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def get_chrome_driver(headless=True):
    """Creates a standardized Chrome WebDriver instance."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1366,850")
    options.add_argument("--log-level=3")
    
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    return driver
