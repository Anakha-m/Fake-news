"""
Unified Master Test Runner for Veritas Fake News Detector Selenium E2E Tests.

Usage:
    python run_selenium_tests.py               # Runs all tests headlessly
    python run_selenium_tests.py --headed      # Runs all tests with visible Chrome UI
    python run_selenium_tests.py --suite user  # Runs only the User portal tests
"""

import os
import sys
import time
import socket
import argparse
import unittest
import threading
from werkzeug.serving import make_server

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app import app, init_db


class ServerThread(threading.Thread):
    """Runs the Flask development server in a background thread."""
    def __init__(self, app, host='127.0.0.1', port=5000):
        super().__init__()
        self.host = host
        self.port = port
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


def is_port_in_use(port=5000, host='127.0.0.1'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main():
    parser = argparse.ArgumentParser(description="Run Selenium End-to-End Tests for Veritas Fake News Detector")
    parser.add_argument("--headed", action="store_true", help="Launch visible browser window instead of headless mode")
    parser.add_argument("--port", type=int, default=5000, help="Port to run tests against (default: 5000)")
    parser.add_argument("--suite", type=str, choices=["all", "smoke", "user", "writer", "admin", "security"], default="all", help="Test suite to execute")
    args = parser.parse_args()

    os.environ["TEST_SERVER_URL"] = f"http://127.0.0.1:{args.port}"
    if args.headed:
        os.environ["HEADED_SELENIUM"] = "1"
    else:
        os.environ.pop("HEADED_SELENIUM", None)

    print("\n" + "=" * 75)
    print("      VERITAS FAKE NEWS DETECTOR - SELENIUM AUTOMATED TEST SUITE")
    print("=" * 75)
    print(f"Target Server:  http://127.0.0.1:{args.port}")
    print(f"Browser Mode:   {'HEADED (UI Visible)' if args.headed else 'HEADLESS (CI / Terminal)'}")
    print(f"Selected Suite: {args.suite.upper()}")
    print("=" * 75 + "\n")

    # 1. Initialize SQLite Database
    init_db()

    # 2. Check or Launch Local Server
    server_thread = None
    if not is_port_in_use(args.port):
        print(f"[INFO] Starting Flask server on 127.0.0.1:{args.port} for test execution...")
        server_thread = ServerThread(app, host='127.0.0.1', port=args.port)
        server_thread.daemon = True
        server_thread.start()
        time.sleep(1.5)
        print("[INFO] Flask test server is live.\n")
    else:
        print(f"[INFO] Connected to existing server running on port {args.port}.\n")

    # 3. Discover and Build Test Suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if args.suite == "smoke":
        from selenium_tests.test_login import run_smoke_test
        success = run_smoke_test(headless=not args.headed)
        if server_thread:
            server_thread.shutdown()
        sys.exit(0 if success else 1)

    elif args.suite == "user":
        from selenium_tests.test_user_portal import TestUserPortalE2E
        suite.addTests(loader.loadTestsFromTestCase(TestUserPortalE2E))

    elif args.suite == "writer":
        from selenium_tests.test_writer_portal import TestWriterPortalE2E
        suite.addTests(loader.loadTestsFromTestCase(TestWriterPortalE2E))

    elif args.suite == "admin":
        from selenium_tests.test_admin_portal import TestAdminPortalE2E
        suite.addTests(loader.loadTestsFromTestCase(TestAdminPortalE2E))

    elif args.suite == "security":
        from selenium_tests.test_security import TestSecurityAndAccessControlE2E
        suite.addTests(loader.loadTestsFromTestCase(TestSecurityAndAccessControlE2E))

    else:  # all
        from selenium_tests.test_user_portal import TestUserPortalE2E
        from selenium_tests.test_writer_portal import TestWriterPortalE2E
        from selenium_tests.test_admin_portal import TestAdminPortalE2E
        from selenium_tests.test_security import TestSecurityAndAccessControlE2E
        
        suite.addTests(loader.loadTestsFromTestCase(TestUserPortalE2E))
        suite.addTests(loader.loadTestsFromTestCase(TestWriterPortalE2E))
        suite.addTests(loader.loadTestsFromTestCase(TestAdminPortalE2E))
        suite.addTests(loader.loadTestsFromTestCase(TestSecurityAndAccessControlE2E))

    # 4. Run Test Suite
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time

    # 5. Summary
    print("\n" + "=" * 75)
    print("SELENIUM TEST EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Passed:          {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:        {len(result.failures)}")
    print(f"Errors:          {len(result.errors)}")
    print(f"Execution Time:  {duration:.2f} seconds")
    print("=" * 75 + "\n")

    if server_thread:
        print("[INFO] Shutting down background test server...")
        server_thread.shutdown()

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
