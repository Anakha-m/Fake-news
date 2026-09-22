"""
Views for Detector Application.
Handles authentication, dashboard rendering, dual-engine prediction execution,
live news evidence retrieval, fact-checking, and prediction history tracking.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.db.models import Q

from .forms import UserRegistrationForm, UserLoginForm, NewsSubmissionForm
from .models import PredictionHistory
from verification.aggregator import verify_news_claim


def login_view(request):
    """Handles user login with username or email."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    next_url = request.GET.get('next') or request.POST.get('next') or 'dashboard'

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username_or_email = form.cleaned_data['username_or_email'].strip()
            password = form.cleaned_data['password']

            # Resolve email to username if needed
            username = username_or_email
            if '@' in username_or_email:
                user_obj = User.objects.filter(email__iexact=username_or_email).first()
                if user_obj:
                    username = user_obj.username

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                return redirect(next_url)
            else:
                messages.error(request, "Invalid username/email or password. Please try again.")
    else:
        form = UserLoginForm()

    return render(request, 'login.html', {'form': form, 'next': next_url})


def register_view(request):
    """
    Handles user registration.
    Redirects to login page upon success so the user explicitly logs in.
    """
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, f"Account '{user.username}' created successfully! Please log in with your credentials to access the system.")
            return redirect('login')
        else:
            messages.error(request, "Please correct the validation errors below to register.")
    else:
        form = UserRegistrationForm()

    return render(request, 'register.html', {'form': form})


@login_required(login_url='login')
def logout_view(request):
    """Logs out user and redirects to login page."""
    logout(request)
    messages.info(request, "You have been successfully logged out.")
    return redirect('login')


@login_required(login_url='login')
def dashboard_view(request):
    """Renders the main fake news detection dashboard."""
    form = NewsSubmissionForm()
    recent_predictions = PredictionHistory.objects.filter(user=request.user)[:5]
    
    return render(request, 'dashboard.html', {
        'form': form,
        'recent_predictions': recent_predictions
    })


@login_required(login_url='login')
def predict_view(request):
    """
    Executes the full Dual-Engine verification pipeline:
    1. Machine Learning Stylistic Prediction
    2. Claim Extraction & Breaking News Detection
    3. Live News Retrieval across APIs
    4. Fact-Checking Registry Verification
    5. Semantic Stance & Source Reliability Analysis
    6. Transparent Decision Synthesis
    """
    if request.method != 'POST':
        return redirect('dashboard')

    form = NewsSubmissionForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Please provide a valid news headline or article (at least 3 words).")
        return render(request, 'dashboard.html', {'form': form})

    news_title = form.cleaned_data.get('news_title', '').strip()
    news_text = form.cleaned_data['news_text'].strip()
    skip_live = form.cleaned_data.get('skip_live_verification', False)

    # Derive headline preview if title is empty
    if not news_title:
        lines = [l.strip() for l in news_text.split('\n') if l.strip()]
        news_title = lines[0][:200] if lines else news_text[:200]

    # Execute Dual-Engine Verification
    try:
        pipeline_result = verify_news_claim(
            title=news_title,
            text=news_text,
            skip_live_verification=skip_live
        )
    except Exception as e:
        messages.error(request, f"Verification pipeline error: {str(e)}")
        return redirect('dashboard')

    if not pipeline_result.get('success'):
        messages.error(request, pipeline_result.get('error', 'Error analyzing news.'))
        return redirect('dashboard')

    ml_res = pipeline_result['ml_analysis']
    claim_info = pipeline_result['claim_info']
    live_news_raw = pipeline_result['live_news_raw']
    fact_check_res = pipeline_result['fact_check_result']
    evidence_res = pipeline_result['evidence_analysis']
    final_assessment = pipeline_result['final_assessment']

    # Extract clean display metrics
    ml_pred = ml_res.get('prediction', 'REAL')
    ml_conf = ml_res.get('confidence_score', ml_res.get('confidence', 50.0))
    if isinstance(ml_conf, float) and ml_conf <= 1.0:
        ml_conf = round(ml_conf * 100, 1)

    real_prob = ml_res.get('real_probability', 0.5)
    fake_prob = ml_res.get('fake_probability', 0.5)

    ev_status = evidence_res.get('evidence_status', 'INSUFFICIENT')
    ev_summary = evidence_res.get('summary', '')
    analyzed_sources = evidence_res.get('analyzed_articles', [])
    
    fc_found = fact_check_res.get('found', False)
    fc_rating = fact_check_res.get('rating', 'Not Available')
    
    verdict = final_assessment.get('final_assessment', 'UNVERIFIED')
    badge_class = final_assessment.get('badge_class', 'secondary')
    explanation = final_assessment.get('explanation', '')
    is_current = claim_info.get('is_current_news', False)

    # Save to user's database history
    history_record = PredictionHistory.objects.create(
        user=request.user,
        news_title=news_title,
        news_text=news_text,
        ml_prediction=ml_pred,
        ml_confidence=ml_conf,
        real_probability=real_prob,
        fake_probability=fake_prob,
        evidence_status=ev_status,
        evidence_summary=ev_summary,
        retrieved_sources=analyzed_sources,
        fact_check_status=fc_rating if fc_found else 'Not Available',
        is_current_news=is_current,
        final_verdict=verdict,
        badge_class=badge_class,
        explanation=explanation
    )

    context = {
        'history_id': history_record.id,
        'news_title': news_title,
        'news_text': news_text,
        'claim_info': claim_info,
        'ml_analysis': ml_res,
        'ml_prediction': ml_pred,
        'ml_confidence': ml_conf,
        'real_probability': round(real_prob * 100, 1) if real_prob <= 1.0 else real_prob,
        'fake_probability': round(fake_prob * 100, 1) if fake_prob <= 1.0 else fake_prob,
        'is_uncertain': ml_res.get('is_uncertain', False),
        'live_news_raw': live_news_raw,
        'fact_check_result': fact_check_res,
        'evidence_analysis': evidence_res,
        'evidence_status': ev_status,
        'evidence_summary': ev_summary,
        'sources': analyzed_sources,
        'final_assessment': final_assessment,
        'final_verdict': verdict,
        'badge_class': badge_class,
        'explanation': explanation,
        'created_at': history_record.created_at
    }

    return render(request, 'result.html', context)


@login_required(login_url='login')
def history_view(request):
    """Displays list of user's past predictions with search and filtering."""
    query = request.GET.get('q', '').strip()
    status_filter = request.GET.get('status', '').strip()

    predictions = PredictionHistory.objects.filter(user=request.user)

    if query:
        predictions = predictions.filter(
            Q(news_text__icontains=query) | 
            Q(news_title__icontains=query) |
            Q(final_verdict__icontains=query)
        )

    if status_filter:
        predictions = predictions.filter(final_verdict__icontains=status_filter)

    return render(request, 'history.html', {
        'predictions': predictions,
        'query': query,
        'status_filter': status_filter,
        'total_count': predictions.count()
    })


@login_required(login_url='login')
def result_detail_view(request, pk):
    """Displays detailed breakdown of a previously saved prediction."""
    record = get_object_or_404(PredictionHistory, pk=pk, user=request.user)

    context = {
        'history_id': record.id,
        'news_title': record.news_title,
        'news_text': record.news_text,
        'ml_prediction': record.ml_prediction,
        'ml_confidence': record.ml_confidence,
        'real_probability': round(record.real_probability * 100, 1) if record.real_probability <= 1.0 else record.real_probability,
        'fake_probability': round(record.fake_probability * 100, 1) if record.fake_probability <= 1.0 else record.fake_probability,
        'evidence_status': record.evidence_status,
        'evidence_summary': record.evidence_summary,
        'sources': record.retrieved_sources,
        'fact_check_result': {
            'found': record.fact_check_status not in ('Not Available', 'Not Checked', ''),
            'rating': record.fact_check_status
        },
        'final_verdict': record.final_verdict,
        'badge_class': record.badge_class or 'secondary',
        'explanation': record.explanation,
        'created_at': record.created_at,
        'is_history_view': True
    }

    return render(request, 'result.html', context)


@login_required(login_url='login')
def clear_history_view(request):
    """Clears all prediction history for the current user."""
    if request.method == 'POST':
        deleted_count, _ = PredictionHistory.objects.filter(user=request.user).delete()
        messages.success(request, f"Successfully cleared {deleted_count} past prediction record(s).")
    return redirect('history')
