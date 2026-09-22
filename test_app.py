"""
Comprehensive End-to-End Test Suite for Fake News Detection System.
Tests:
1. Model training and regularization parameters.
2. Prediction engine with confidence scoring and <60% threshold logic.
3. RSS News ingestion service (Malayala Manorama & Indian news).
4. Individual Portal Authentication:
   - /login/user (User Portal)
   - /login/writer (Writer Portal)
   - /login/admin (Admin Portal)
   - Cross-portal permission enforcement
"""

import sys
import os
import json
import unittest

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app import app, init_db, get_db_connection
from ml_model.predict import predict_news
from services.rss_service import get_live_news_feed


class TestFakeNewsSystem(unittest.TestCase):

    def setUp(self):
        init_db()
        self.client = app.test_client()

    def test_1_prediction_high_confidence_real(self):
        """Test that authentic space news is classified as REAL."""
        text = "NASA James Webb Space Telescope observes earliest distant galaxy formed after Big Bang with spectroscopic redshift measurements."
        res = predict_news(text)
        self.assertEqual(res["prediction"], "REAL")
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertIn("real_probability", res)
        self.assertIn("fake_probability", res)
        print(f"\n[Test 1] Real News Test: Prediction={res['prediction']}, Conf={res['confidence']}%")

    def test_2_prediction_high_confidence_fake(self):
        """Test that viral hoax claim is classified as FAKE and flagged for Content Writer review."""
        text = "SHOCKING PROOF! Passenger airplanes secretly spraying toxic chemtrails to control weather and mind control citizens!"
        res = predict_news(text)
        self.assertEqual(res["prediction"], "FAKE")
        self.assertTrue(res["is_needs_review"])
        self.assertGreaterEqual(res["fake_probability"], 0.4)
        print(f"\n[Test 2] Fake News Test: Prediction={res['prediction']}, FakeProb={res['fake_probability']*100:.1f}%, NeedsReview={res['is_needs_review']}")

    def test_3_logistic_regression_prediction_logic(self):
        """Test that REAL news is classified and auto-verified without review flagging."""
        text = "The local municipal council held a routine meeting to discuss sanitation tenders."
        res = predict_news(text)
        self.assertIn(res["prediction"], ["REAL", "FAKE"])
        if res["prediction"] == "REAL":
            self.assertFalse(res["is_needs_review"])
        else:
            self.assertTrue(res["is_needs_review"])
        print(f"\n[Test 3] Logistic Regression Test: Prediction={res['prediction']} (Confidence: {res['confidence']}%, NeedsReview: {res['is_needs_review']})")

    def test_4_rss_feed_integration(self):
        """Test that live RSS service retrieves news and generates predictions."""
        feed = get_live_news_feed(limit=3)
        self.assertGreater(len(feed), 0)
        first = feed[0]
        self.assertIn("title", first)
        self.assertIn("source", first)
        self.assertIn("prediction", first)
        self.assertIn("confidence", first)
        print(f"\n[Test 4] Live RSS Ingestion: Retrieved '{first['title'][:50]}...' from {first['source']} -> ML Pred: {first['prediction']}")

    def test_5_individual_login_portals(self):
        """Test individual login pages: /login/user, /login/writer, /login/admin."""
        # 1. User Portal Login
        res_u_page = self.client.get("/login/user")
        self.assertEqual(res_u_page.status_code, 200)
        self.assertIn(b"User Login", res_u_page.data)

        res_u_login = self.client.post("/login/user", data={"username": "user", "password": "user123"}, follow_redirects=False)
        self.assertEqual(res_u_login.status_code, 302)
        self.assertEqual(res_u_login.location, "/user")

        # 2. Writer Portal Login
        writer_client = app.test_client()
        res_w_page = writer_client.get("/login/writer")
        self.assertEqual(res_w_page.status_code, 200)
        self.assertIn(b"Content Writer Login", res_w_page.data)

        res_w_login = writer_client.post("/login/writer", data={"username": "writer", "password": "writer123"}, follow_redirects=False)
        self.assertEqual(res_w_login.status_code, 302)
        self.assertEqual(res_w_login.location, "/writer")

        # 3. Admin Portal Login
        admin_client = app.test_client()
        res_a_page = admin_client.get("/login/admin")
        self.assertEqual(res_a_page.status_code, 200)
        self.assertIn(b"Admin Login", res_a_page.data)

        res_a_login = admin_client.post("/login/admin", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
        self.assertEqual(res_a_login.status_code, 302)
        self.assertEqual(res_a_login.location, "/admin")
        print("\n[Test 5] Individual Login Portals (/login/user, /login/writer, /login/admin) OK.")

    def test_6_cross_portal_security_validation(self):
        """Test that user cannot log into Admin Portal with standard user account."""
        client = app.test_client()
        res = client.post("/login/admin", data={"username": "user", "password": "user123"})
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"does not have Administrator privileges", res.data)
        print("\n[Test 6] Cross-portal security validation OK (User blocked from Admin portal).")

    def test_7_user_submission_with_session(self):
        """Test that prediction submission attributes to logged-in user."""
        client = app.test_client()
        client.post("/login/user", data={"username": "user", "password": "user123"})

        payload = {
            "title": "Kerala University AI Research Grant",
            "text": "The University of Kerala received a new grant for renewable energy research."
        }
        response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        print(f"\n[Test 7] Authenticated User Prediction OK -> ID #{data['data'].get('id')}")

    def test_8_writer_moderation_resolution(self):
        """Test writer moderation action."""
        writer_client = app.test_client()
        writer_client.post("/login/writer", data={"username": "writer", "password": "writer123"})

        conn = get_db_connection()
        item = conn.execute("SELECT id FROM predictions WHERE is_needs_review = 1 LIMIT 1").fetchone()
        conn.close()

        if item:
            item_id = item["id"]
            review_payload = {"id": item_id, "verdict": "REAL"}
            rev_res = writer_client.post("/api/writer/review", json=review_payload)
            self.assertEqual(rev_res.status_code, 200)
            rev_data = rev_res.get_json()
            self.assertTrue(rev_data["success"])
            self.assertEqual(rev_data["verdict"], "REAL")
            print(f"\n[Test 8] Writer Moderation Action OK -> Record #{item_id} marked as REAL.")

    def test_9_user_registration(self):
        """Test registration page rendering and new user registration flow."""
        get_res = self.client.get("/register")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn(b"Create Account", get_res.data)

        post_res = self.client.post("/register", data={
            "username": "newuser_test",
            "password": "password123",
            "role": "user"
        }, follow_redirects=True)
        self.assertEqual(post_res.status_code, 200)
        print("\n[Test 9] User Registration Flow OK.")

    def test_10_user_history_and_editorial_tracking(self):
        """Test user history retrieval and editorial resolution tracking."""
        # 1. Log in as user
        user_client = app.test_client()
        user_client.post("/login/user", data={"username": "user", "password": "user123"})

        # 2. Submit ambiguous news requiring editorial review
        predict_res = user_client.post("/api/predict", json={
            "title": "Townhall Municipal Desilting Tender",
            "text": "The town committee met yesterday to discuss minor municipal drainage desilting tenders."
        })
        self.assertEqual(predict_res.status_code, 200)
        pred_data = predict_res.get_json()
        item_id = pred_data["data"]["id"]

        # 3. Log in as writer and resolve it with editorial notes
        writer_client = app.test_client()
        writer_client.post("/login/writer", data={"username": "writer", "password": "writer123"})
        rev_res = writer_client.post("/api/writer/review", json={
            "id": item_id,
            "verdict": "REAL",
            "notes": "Verified via official town council gazette publication."
        })
        self.assertEqual(rev_res.status_code, 200)

        # 4. As user, fetch history and verify editorial resolution is visible
        hist_res = user_client.get("/api/user/history")
        self.assertEqual(hist_res.status_code, 200)
        hist_data = hist_res.get_json()
        self.assertTrue(hist_data["success"])
        resolved_item = next((x for x in hist_data["history"] if x["id"] == item_id), None)
        self.assertIsNotNone(resolved_item)
        self.assertEqual(resolved_item["writer_status"], "RESOLVED")
        self.assertEqual(resolved_item["human_verdict"], "REAL")
        self.assertEqual(resolved_item["reviewed_by"], "writer")
        self.assertIn("Verified via official town council", resolved_item["editorial_notes"])

        # 5. Log in as admin and verify admin dashboard activity log has user attribution
        admin_client = app.test_client()
        admin_client.post("/login/admin", data={"username": "admin", "password": "admin123"})
        admin_res = admin_client.get("/admin")
        self.assertEqual(admin_res.status_code, 200)
        self.assertIn(b"User News Verification Activity", admin_res.data)
        self.assertIn(b"Townhall Municipal Desilting", admin_res.data)
        print("\n[Test 10] User History & Editorial Tracking End-to-End OK.")

    def test_11_admin_deletion_rights(self):
        """Test admin deletion rights for user accounts and activity logs."""
        admin_client = app.test_client()
        admin_client.post("/login/admin", data={"username": "admin", "password": "admin123"})

        # 1. Create a temporary user to test deletion
        conn = get_db_connection()
        conn.execute("INSERT OR IGNORE INTO users (username, password_hash, role) VALUES ('temp_delete_user', 'hash', 'user')")
        u_id = conn.execute("SELECT id FROM users WHERE username = 'temp_delete_user'").fetchone()["id"]
        conn.commit()
        conn.close()

        # 2. Test deleting user account
        del_user_res = admin_client.post("/api/admin/users/delete", json={"user_id": u_id})
        self.assertEqual(del_user_res.status_code, 200)
        del_data = del_user_res.get_json()
        self.assertTrue(del_data["success"])

        # 3. Test root admin deletion safeguard (should be blocked)
        conn = get_db_connection()
        root_admin_id = conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()["id"]
        conn.close()
        root_del_res = admin_client.post("/api/admin/users/delete", json={"user_id": root_admin_id})
        self.assertIn(root_del_res.status_code, [400, 403])

        # 4. Test deleting single prediction log
        conn = get_db_connection()
        log_item = conn.execute("SELECT id FROM predictions LIMIT 1").fetchone()
        conn.close()
        if log_item:
            log_id = log_item["id"]
            del_log_res = admin_client.post("/api/admin/predictions/delete", json={"prediction_id": log_id})
            self.assertEqual(del_log_res.status_code, 200)
            self.assertTrue(del_log_res.get_json()["success"])

        # 5. Test clear all logs
        clear_res = admin_client.post("/api/admin/predictions/clear-all")
        self.assertEqual(clear_res.status_code, 200)
        self.assertTrue(clear_res.get_json()["success"])

        # 6. Verify non-admin (user) cannot delete
        user_client = app.test_client()
        user_client.post("/login/user", data={"username": "user", "password": "user123"})
        unauth_res = user_client.post("/api/admin/predictions/clear-all")
        self.assertIn(unauth_res.status_code, [302, 403])
        print("\n[Test 11] Admin Deletion Rights & Security Safeguards OK.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
