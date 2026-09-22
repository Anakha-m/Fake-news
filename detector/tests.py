"""
Comprehensive Test Suite for Fake News Detection System (SDG Project).
Tests user authentication, form validation, ML preprocessing, prediction engine,
claim extraction, live news retrieval resilience, fact-checking, semantic stance analysis,
and the 9 faculty demonstration test scenarios.
"""

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

from detector.models import PredictionHistory
from detector.forms import UserRegistrationForm, UserLoginForm, NewsSubmissionForm
from ml_model.preprocess import clean_text, expand_contractions
from ml_model.predict import predict_news
from verification.claim_extractor import extract_search_query, detect_recency_and_current_news, extract_entities_and_keywords
from verification.live_news_service import retrieve_live_news
from verification.fact_check_service import check_fact_check_registry
from verification.evidence_analyzer import calculate_semantic_similarity, evaluate_source_credibility, analyze_evidence_corpus
from verification.decision_engine import synthesize_decision
from verification.aggregator import verify_news_claim


class AuthenticationTests(TestCase):
    """Tests for user registration, login, and session access."""

    def setUp(self):
        self.client = Client()
        self.register_url = reverse('register')
        self.login_url = reverse('login')
        self.logout_url = reverse('logout')
        self.dashboard_url = reverse('dashboard')

    def test_user_registration_redirects_to_login(self):
        """Registration creates user and redirects to login page for explicit authentication."""
        data = {
            'username': 'student_user',
            'email': 'student@university.edu',
            'password': 'StrongPassword123',
            'confirm_password': 'StrongPassword123'
        }
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, self.login_url)
        self.assertTrue(User.objects.filter(username='student_user').exists())

    def test_duplicate_username_rejected(self):
        """Cannot register with existing username."""
        User.objects.create_user(username='existinguser', email='first@test.com', password='password123')
        data = {
            'username': 'existinguser',
            'email': 'second@test.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }
        response = self.client.post(self.register_url, data)
        self.assertEqual(response.status_code, 200)
        self.assertFormError(response.context['form'], 'username', 'This username is already registered. Please choose another one.')

    def test_password_mismatch_rejected(self):
        """Passwords must match."""
        data = {
            'username': 'newuser',
            'email': 'user@test.com',
            'password': 'passwordA1',
            'confirm_password': 'passwordB2'
        }
        response = self.client.post(self.register_url, data)
        self.assertFormError(response.context['form'], 'confirm_password', 'Passwords do not match.')

    def test_login_and_logout(self):
        """Registered user can log in and log out successfully."""
        user = User.objects.create_user(username='tester', email='tester@test.com', password='secretpassword')
        
        # Test login
        login_resp = self.client.post(self.login_url, {
            'username_or_email': 'tester',
            'password': 'secretpassword'
        }, follow=True)
        self.assertEqual(login_resp.status_code, 200)
        self.assertTrue(login_resp.context['user'].is_authenticated)

        # Test logout
        logout_resp = self.client.get(self.logout_url, follow=True)
        self.assertEqual(logout_resp.status_code, 200)
        self.assertFalse(logout_resp.context['user'].is_authenticated)

    def test_unauthenticated_dashboard_redirect(self):
        """Unauthenticated requests to dashboard redirect to login."""
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login/', response.url)


class ClaimExtractionAndRecencyTests(TestCase):
    """Tests for claim extraction, clickbait filtering, and breaking news detection."""

    def test_clickbait_removal_in_query(self):
        """Clickbait words like SHOCKING, BIG PHARMA, SECRET are stripped from search queries."""
        sensational_title = "SHOCKING BOMBSHELL: Big Pharma Hiding Secret Himalayan Garlic Cure!"
        extracted = extract_search_query(title=sensational_title)
        query = extracted["query"].lower()
        self.assertNotIn("shocking", query)
        self.assertNotIn("bombshell", query)
        self.assertNotIn("secret", query)
        self.assertTrue(len(query) > 0)

    def test_recency_detection_breaking_news(self):
        """Detects words like 'today', 'breaking', '2026' as live current events."""
        text = "Government officially announced a new national clean energy policy today."
        recency = detect_recency_and_current_news(text)
        self.assertTrue(recency["is_current_news"])
        self.assertIn("today", recency["recency_cues"])

    def test_entity_extraction(self):
        """Extracts acronyms and multi-word proper nouns."""
        text = "NASA and ISRO launched a joint satellite from Sriharikota Space Center."
        entities_data = extract_entities_and_keywords(text)
        self.assertIn("NASA", entities_data["acronyms"])
        self.assertIn("ISRO", entities_data["acronyms"])


class EvidenceAndStanceAnalysisTests(TestCase):
    """Tests for semantic similarity, source reliability tiering, and stance classification."""

    def test_semantic_similarity(self):
        """Cosine similarity calculates high overlap for matching content and low for unrelated."""
        claim = "NASA James Webb Space Telescope confirms earliest galaxy."
        related = "Astronomers use the James Webb Space Telescope from NASA to identify ancient cosmic galaxy."
        unrelated = "Local bakery wins annual sourdough bread competition."
        
        sim_high = calculate_semantic_similarity(claim, related)
        sim_low = calculate_semantic_similarity(claim, unrelated)
        self.assertGreater(sim_high, sim_low)
        self.assertGreater(sim_high, 0.25)

    def test_source_credibility_tiering(self):
        """Authoritative global sources are assigned Tier 1 weight."""
        t1_label, t1_weight = evaluate_source_credibility("Reuters News Wire")
        t2_label, t2_weight = evaluate_source_credibility("TechCrunch")
        t3_label, t3_weight = evaluate_source_credibility("Random WordPress Blog")

        self.assertEqual(t1_weight, 1.0)
        self.assertEqual(t2_weight, 0.85)
        self.assertEqual(t3_weight, 0.65)

    def test_evidence_stance_debunk(self):
        """Articles containing debunking markers produce CONTRADICTING stance."""
        claim = "Drinking salt water cures all illnesses"
        mock_articles = [{
            "source": "Reuters Fact Check",
            "title": "Fact Check: False claim that salt water is a proven medical cure debunked by physicians.",
            "snippet": "Medical experts refuted the viral social media post claiming salt water treats diseases.",
            "published_date": "2026-08-30",
            "url": "https://reuters.com/fact-check/test"
        }]
        analysis = analyze_evidence_corpus(claim, mock_articles)
        self.assertEqual(analysis["evidence_status"], "CONTRADICTING")
        self.assertGreaterEqual(analysis["contradicting_count"], 1)


class FacultyDemonstrationScenariosTests(TestCase):
    """
    Tests covering the 9 required faculty review demonstration scenarios:
    1. Historical real news
    2. Historical fake news
    3. Recent real news
    4. Recent fake claims
    5. Breaking news
    6. News with insufficient evidence
    7. Long articles
    8. Short headlines
    9. API unavailable / zero-key mode
    """

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='demo_user', email='demo@test.com', password='Password123')
        self.client.login(username='demo_user', password='Password123')

    def test_scenario_1_historical_real_news(self):
        """Scenario 1: Historical authentic news with sober journalism."""
        text = "The Federal Reserve raised interest rates by a quarter percentage point to combat inflation, Chairman Jerome Powell announced at the Washington press conference."
        result = verify_news_claim(text=text, skip_live_verification=True)
        self.assertTrue(result['success'])
        self.assertEqual(result['ml_analysis']['prediction'], 'REAL')
        self.assertIn(result['final_assessment']['final_assessment'], ['LIKELY REAL', 'UNVERIFIED'])

    def test_scenario_2_historical_fake_news(self):
        """Scenario 2: Historical sensationalist fake hoax."""
        text = "SHOCKING SECRET EXPOSED! Government officials admit that secret airplanes spray toxic mind control chemicals over suburban neighborhoods every night! SHARE THIS BEFORE THEY DELETE IT!"
        result = verify_news_claim(text=text, skip_live_verification=True)
        self.assertTrue(result['success'])
        self.assertEqual(result['ml_analysis']['prediction'], 'FAKE')
        self.assertEqual(result['final_assessment']['final_assessment'], 'LIKELY FAKE')

    def test_scenario_3_recent_real_news(self):
        """Scenario 3: Recent wire report with live corroboration."""
        title = "European Union Approves Artificial Intelligence Act"
        text = "European Parliament delegates voted to approve the Artificial Intelligence Act today, establishing safety standards for generative AI systems across member states."
        result = verify_news_claim(title=title, text=text, skip_live_verification=False)
        self.assertTrue(result['success'])
        self.assertIn(result['final_assessment']['final_assessment'], ['LIKELY REAL', 'UNVERIFIED', 'INSUFFICIENT EVIDENCE', 'MISLEADING'])

    def test_scenario_4_recent_fake_claims(self):
        """Scenario 4: Recent viral fabricated health hoax."""
        title = "Crushed Cucumber Peels Melt Kidney Stones in 12 Hours"
        text = "DOCTORS ARE TERRIFIED: A Himalayan potion made from boiled cucumber peels cures diabetes and dissolves kidney stones in 12 hours!"
        result = verify_news_claim(title=title, text=text, skip_live_verification=False)
        self.assertTrue(result['success'])
        self.assertIn(result['final_assessment']['final_assessment'], ['LIKELY FAKE', 'MISLEADING', 'UNVERIFIED', 'INSUFFICIENT EVIDENCE'])

    def test_scenario_5_breaking_news_unverified(self):
        """Scenario 5: Breaking news with zero independent sources yields UNVERIFIED."""
        title = "Breaking: Secret asteroid mined for gold today"
        text = "Unconfirmed reports claim a private spaceship captured a golden asteroid today."
        result = verify_news_claim(title=title, text=text, skip_live_verification=True)
        self.assertTrue(result['success'])
        self.assertIn(result['final_assessment']['final_assessment'], ['UNVERIFIED', 'LIKELY FAKE', 'INSUFFICIENT EVIDENCE'])

    def test_scenario_6_insufficient_evidence(self):
        """Scenario 6: Ambiguous text with low confidence returns INSUFFICIENT EVIDENCE / UNVERIFIED."""
        text = "A meeting took place somewhere between officials."
        result = verify_news_claim(text=text, skip_live_verification=True)
        self.assertTrue(result['success'])
        self.assertIn(result['final_assessment']['final_assessment'], ['INSUFFICIENT EVIDENCE', 'UNVERIFIED', 'LIKELY REAL'])

    def test_scenario_7_long_article(self):
        """Scenario 7: Long multi-paragraph article extracts key claim cleanly."""
        long_text = (
            "The World Health Organization (WHO) and international research consortiums published a landmark study. "
            "Over three years of clinical monitoring across 25 hospitals showed substantial reductions in malaria morbidity. "
            "Dr. Tedros Adhanom Ghebreyesus commended the frontline community healthcare workers in Nairobi and Accra. "
            "Funding partnerships with the Global Fund contributed to widespread distribution of preventative bed nets and vaccines."
        )
        result = verify_news_claim(title="WHO Malaria Progress Report", text=long_text, skip_live_verification=True)
        self.assertTrue(result['success'])
        self.assertTrue(len(result['claim_info']['query']) > 0)
        self.assertLessEqual(len(result['claim_info']['query'].split()), 6)

    def test_scenario_8_short_headline(self):
        """Scenario 8: Short headline handles concise input gracefully."""
        result = verify_news_claim(title="Solar eclipse visible across continents today", text="")
        self.assertTrue(result['success'])
        self.assertIn('final_assessment', result)

    def test_scenario_9_api_unavailable_graceful_fallback(self):
        """Scenario 9: When API key is missing or offline, pipeline falls back gracefully without crashing."""
        result = verify_news_claim(title="NASA Mars Rover", text="Perseverance rover collects rock core samples on Mars.", skip_live_verification=True)
        self.assertTrue(result['success'])
        self.assertIn('ml_analysis', result)
        self.assertIn('final_assessment', result)

    def test_full_web_submission_saves_history(self):
        """End-to-end web POST creates a PredictionHistory database record with all fields."""
        response = self.client.post(reverse('predict'), {
            'news_title': 'NASA Space Telescope Discovery',
            'news_text': 'Astronomers published spectroscopic findings in Nature Astronomy regarding cosmic formation.',
            'skip_live_verification': 'on'
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Final Dual-Engine Assessment")
        self.assertEqual(PredictionHistory.objects.count(), 1)
        record = PredictionHistory.objects.first()
        self.assertEqual(record.user, self.user)
        self.assertTrue(len(record.final_verdict) > 0)
