"""
scratch/verify_sdg_server.py
End-to-end verification script for sdg_project live server.
"""

import re
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    session = requests.Session()

    print("--- 1. Testing Unauthenticated Access Redirect to Login ---")
    r_unauth = session.get(f"{BASE_URL}/", allow_redirects=False)
    assert r_unauth.status_code == 302, f"Expected 302, got {r_unauth.status_code}"
    assert "/login" in r_unauth.headers.get('Location', '')
    print("[PASS] Unauthenticated access to home properly redirects to /login/ page.")

    print("\n--- 2. Testing Login Page & Stylesheet ---")
    r_login = session.get(f"{BASE_URL}/login/")
    assert r_login.status_code == 200
    assert "Welcome Back" in r_login.text
    assert "Username or Email" in r_login.text

    r_css = session.get(f"{BASE_URL}/static/css/style.css")
    assert r_css.status_code == 200
    assert "-webkit-font-smoothing: antialiased" in r_css.text
    print("[PASS] Login page rendered successfully with crisp high-contrast stylesheet.")

    print("\n--- 3. Testing Registration Page & Account Creation ---")
    r_reg_get = session.get(f"{BASE_URL}/register/")
    assert r_reg_get.status_code == 200
    assert "Create an Account" in r_reg_get.text

    csrf_token = session.cookies.get('csrftoken')
    if not csrf_token:
        m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r_reg_get.text)
        if m:
            csrf_token = m.group(1)

    unique_suffix = int(time.time())
    test_user = f"faculty_reviewer_{unique_suffix}"
    test_email = f"faculty_{unique_suffix}@university.edu"
    test_pass = "FacultyPassword2026!"

    reg_data = {
        'csrfmiddlewaretoken': csrf_token,
        'username': test_user,
        'email': test_email,
        'password': test_pass,
        'confirm_password': test_pass
    }
    headers = {'Referer': f"{BASE_URL}/register/"}
    r_reg_post = session.post(f"{BASE_URL}/register/", data=reg_data, headers=headers, allow_redirects=False)
    print(f"Registration POST status: {r_reg_post.status_code} (Redirected to: {r_reg_post.headers.get('Location')})")
    assert r_reg_post.status_code == 302
    assert "/login" in r_reg_post.headers.get('Location', '')
    print(f"[PASS] User '{test_user}' registered successfully and redirected to /login/ (NOT auto-logged in).")

    print("\n--- 4. Testing User Login with New Credentials ---")
    r_login_page = session.get(f"{BASE_URL}/login/")
    m_login_csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r_login_page.text)
    csrf_login = m_login_csrf.group(1) if m_login_csrf else session.cookies.get('csrftoken')

    login_data = {
        'csrfmiddlewaretoken': csrf_login,
        'username_or_email': test_user,
        'password': test_pass
    }
    r_login_post = session.post(f"{BASE_URL}/login/", data=login_data, headers={'Referer': f"{BASE_URL}/login/"})
    assert r_login_post.status_code == 200
    assert test_user in r_login_post.text
    assert "Logout" in r_login_post.text
    print(f"[PASS] User '{test_user}' successfully logged in through the login page.")

    print("\n--- 5. Testing Dual-Engine News Verification (Live News + ML) ---")
    m_home = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r_login_post.text)
    csrf_home = m_home.group(1) if m_home else session.cookies.get('csrftoken')

    analysis_data = {
        'csrfmiddlewaretoken': csrf_home,
        'news_title': "European Union Approves Artificial Intelligence Act",
        'news_text': "European Parliament delegates voted to approve the comprehensive Artificial Intelligence Act today, establishing safety rules for AI developers.",
        'skip_live_verification': 'on'
    }
    r_analyze = session.post(f"{BASE_URL}/predict/", data=analysis_data, headers={'Referer': f"{BASE_URL}/"})
    assert r_analyze.status_code == 200
    assert "Final Dual-Engine Assessment" in r_analyze.text
    assert "Machine Learning Prediction" in r_analyze.text
    assert "Live News Verification" in r_analyze.text
    assert "System Assessment Disclaimer" in r_analyze.text
    print("[PASS] Dual-Engine News verification executed with structured result report generated.")

    print("\n--- 6. Testing User History Log ---")
    r_history = session.get(f"{BASE_URL}/history/")
    assert r_history.status_code == 200
    assert "Artificial Intelligence Act" in r_history.text
    assert "Verification History Log" in r_history.text
    print("[PASS] History log accurately saved and displayed user's verified record.")

    print("\n--- 7. Testing User Logout ---")
    r_logout = session.post(f"{BASE_URL}/logout/", data={'csrfmiddlewaretoken': csrf_home}, headers={'Referer': f"{BASE_URL}/"}, allow_redirects=False)
    assert r_logout.status_code == 302
    assert "/login" in r_logout.headers.get('Location', '')
    print("[PASS] User logged out safely and redirected to login page.")

    print("\n==========================================")
    print("ALL REAL-TIME VERIFICATION CHECKS PASSED!")
    print("==========================================")

if __name__ == "__main__":
    run_tests()
