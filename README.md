# FAKE NEWS DETECTION SYSTEM
### Dual-Engine Machine Learning & Real-Time News Verification System

**Project Name:** `Fake News Detection System`  
**Workspace Folder:** `sdg_project`

---

## 1. Project Overview & Motivation

When evaluating breaking news (e.g., *"India has announced a new nationwide policy"* or *"Astronomers detect new signals from deep space"*), traditional NLP machine learning models face a critical limitation:
> **"How will a statistical model trained on historical data verify whether an ongoing, live news event from today is real or fake?"**

The **Fake News Detection System** resolves this fundamental limitation by implementing a **Dual-Engine Architecture**:
1. **Engine 1: Supervised ML Stylistic Classifier:** Analyzes linguistic patterns, sensationalism markers, rhetoric, capitalization, and emotional framing trained on verified datasets.
2. **Engine 2: Real-Time Live News & Fact-Check Verification Engine:** Extracts key entities and claims, queries live news feeds (NewsAPI / GNews / DuckDuckGo News / Wikipedia) and verified fact-checking indexes (Google Fact Check Tools / ClaimReview / Snopes / PolitiFact / BoomLive), evaluates source reliability tiers, and analyzes semantic stance (`SUPPORT`, `CONTRADICT`, `UNVERIFIED`).
3. **Combined Decision System:** Synthesizes ML confidence and external evidence into transparent, explainable verdicts: **LIKELY REAL**, **LIKELY FAKE**, **MISLEADING**, **UNVERIFIED**, or **INSUFFICIENT EVIDENCE**.

---

## 2. System Architecture

```text
                                [ User Enters News / Headline ]
                                                │
                                                ▼
                                      [ Text Preprocessing ]
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                                                       ▼
      [ Engine 1: ML Model Prediction ]                       [ Engine 2: Claim Extraction ]
                    │                                                       │
         [ TF-IDF Feature Vector ]                           [ Entity / Recency Extraction ]
                    │                                                       │
         [ Logistic Regression ]                           [ Live News Search / Fact Check ]
         (Prediction & Confidence)                         (NewsAPI / GNews / Open Search)
                    │                                                       │
                    │                                         [ Semantic Evidence Analysis ]
                    │                                       (Source Tiers & Stance Stance)
                    │                                                       │
                    └───────────────────────────┬───────────────────────────┘
                                                ▼
                                [ Combined Decision Engine ]
                                (ML + Live Evidence + Fact Check)
                                                │
                                                ▼
                                      [ Final Assessment ]
                        (LIKELY REAL / LIKELY FAKE / MISLEADING / UNVERIFIED)
                                                │
                                                ▼
                                    [ Result UI & History Log ]
```

---

## 3. Real-Time News & Fact-Checking APIs

### A. Supported News APIs
1. **NewsAPI.org (`NEWS_API_KEY`)**:
   - **Why Selected:** Industry standard commercial news indexing API indexing over 80,000 global news sources.
   - **How to Get a Free Key:** Sign up at [newsapi.org/register](https://newsapi.org/register).
   - **Free Tier Limitations:** 100 requests/day, 1-month historical archive, development on localhost.
2. **GNews API (`GNEWS_API_KEY`)**:
   - **Why Selected:** Fast lightweight global news search endpoint.
   - **How to Get a Free Key:** Sign up at [gnews.io](https://gnews.io).
   - **Free Tier Limitations:** 100 requests/day.
3. **DuckDuckGo Open Search & Wikipedia API (Zero-Key Automatic Fallback)**:
   - **Why Selected:** 100% free, zero-configuration fallback requiring NO API keys. Ensures the system never crashes even without internet API keys.

### B. Fact-Check API
- **Google Fact Check Tools API (`GOOGLE_FACT_CHECK_API_KEY`)**:
  - Searches structured ClaimReview fact-checking data from Snopes, PolitiFact, Reuters Fact Check, BOOM Live, FactCheck.org, and AP.
  - Automatically normalizes diverse ratings (*False, Pants on Fire, Misleading, Correct*) into standard verifiable verdicts.

### C. Environment Configuration (`.env`)
Create a `.env` file in the project root:
```ini
# Django Secret Key
SECRET_KEY=django-insecure-sdg-fake-news-detection-system-key-2026-secure
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Live News APIs (Optional - leave blank for zero-key open fallback)
NEWS_API_KEY=
GNEWS_API_KEY=
GOOGLE_FACT_CHECK_API_KEY=
```

---

## 4. Verification Subsystem Modules

- **[`verification/claim_extractor.py`](file:///C:/Users/user/.gemini/antigravity-ide/scratch/sdg_project/verification/claim_extractor.py)**:
  - Strips sensationalist clickbait terms (`SHOCKING`, `BOMBSHELL`, `BIG PHARMA`, `SECRET`).
  - Extracts proper entities, acronyms (NASA, ISRO, WHO, RBI), locations, and organizations.
  - Detects recency cues (`today`, `yesterday`, `breaking`, `currently`, `2026`).
- **[`verification/live_news_service.py`](file:///C:/Users/user/.gemini/antigravity-ide/scratch/sdg_project/verification/live_news_service.py)**:
  - Fetches recent news articles with titles, publication dates, source names, snippets, and URLs.
  - Gracefully handles missing keys, rate limits, timeouts, and network offline conditions.
- **[`verification/fact_check_service.py`](file:///C:/Users/user/.gemini/antigravity-ide/scratch/sdg_project/verification/fact_check_service.py)**:
  - Queries fact-checking registries and identifies if a claim was formally investigated.
- **[`verification/evidence_analyzer.py`](file:///C:/Users/user/.gemini/antigravity-ide/scratch/sdg_project/verification/evidence_analyzer.py)**:
  - Computes semantic cosine similarity and evaluates source credibility across tiers (Tier 1: Global Wires/PIB/Nature, Tier 2: Mainstream News, Tier 3: Web Blogs).
  - Classifies stance into `SUPPORT`, `CONTRADICT`, `UNCLEAR`, `UNRELATED`.
- **[`verification/decision_engine.py`](file:///C:/Users/user/.gemini/antigravity-ide/scratch/sdg_project/verification/decision_engine.py)**:
  - Combines ML and evidence signals into 5 transparent verdicts: **LIKELY REAL**, **LIKELY FAKE**, **MISLEADING**, **UNVERIFIED**, **INSUFFICIENT EVIDENCE**.
- **[`verification/aggregator.py`](file:///C:/Users/user/.gemini/antigravity-ide/scratch/sdg_project/verification/aggregator.py)**:
  - Master coordinator for the entire dual-engine pipeline.

---

## 5. How to Run the Project (Windows Command Prompt)

```cmd
:: 1. Navigate to the project directory
cd /d C:\Users\user\.gemini\antigravity-ide\scratch\sdg_project

:: 2. Apply database migrations
python manage.py migrate

:: 3. (Optional) Run the automated test suite
python manage.py test

:: 4. Start the Django development server
python manage.py runserver
```

Open your browser at: **`http://127.0.0.1:8000/login/`**

---

## 6. Faculty Demonstration Scenarios

| Scenario | Input Example | Expected Behavior |
|---|---|---|
| **1. Historical Real News** | Fed interest rate announcement | ML classifies as REAL; sober journalistic tone confirmed. |
| **2. Historical Fake News** | Secret airplane mind control chemicals | ML detects high sensationalism and conspiracy keywords $\rightarrow$ LIKELY FAKE. |
| **3. Recent Real News** | European Union Approves AI Act | Live search finds corroborating wire reports with publication dates $\rightarrow$ LIKELY REAL. |
| **4. Recent Fake Claim** | Cucumber peels cure diabetes in 12 hours | Debunking / conflict markers identified $\rightarrow$ LIKELY FAKE. |
| **5. Breaking News** | Secret asteroid mined for gold today | Zero independent news reports indexed $\rightarrow$ UNVERIFIED / INSUFFICIENT EVIDENCE. |
| **6. Ambiguous Statement** | Generic meeting between officials | Low confidence with no live evidence $\rightarrow$ INSUFFICIENT EVIDENCE. |
| **7. Long Full Article** | Multi-paragraph WHO report | Claim extractor isolates entities and creates concise search query. |
| **8. Short Headline** | 1-line statement | Handled gracefully with fallback confidence analysis. |
| **9. API Offline / No Key** | Any input | System seamlessly uses DuckDuckGo / Wikipedia open search fallback. |

---

## 8. Selenium Automated End-to-End Testing

The project includes an end-to-end Selenium test suite in `selenium_tests/` testing all three portals (User, Writer, Admin).

### Running Selenium Tests:

```powershell
# 1. Run all tests headlessly (Default / CI):
python run_selenium_tests.py

# 2. Run with visible Chrome UI window:
python run_selenium_tests.py --headed

# 3. Run individual portal suites:
python run_selenium_tests.py --suite smoke     # Fast smoke test
python run_selenium_tests.py --suite user      # User portal test suite
python run_selenium_tests.py --suite writer    # Content Writer moderation suite
python run_selenium_tests.py --suite admin     # Admin dashboard & analytics suite
python run_selenium_tests.py --suite security  # Role-Based Access Control suite
```

---

## 9. Pushing Project to Git / GitHub

To push this project to a new or existing Git repository:

```powershell
# 1. Initialize local repository (if not already done)
git init -b main

# 2. Stage all project files (.gitignore protects secrets and temp files)
git add .

# 3. Create initial commit
git commit -m "Initial commit: Veritas AI Fake News Detection Platform"

# 4. Link your remote repository (replace with your GitHub/GitLab URL)
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 5. Push to remote
git push -u origin main
```

---

## 10. Mandatory Academic Disclaimer

> *"This system combines machine learning predictions with currently available online evidence. The result is an automated assessment and should not be considered absolute proof."*
