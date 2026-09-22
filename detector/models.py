from django.db import models
from django.contrib.auth.models import User

class PredictionHistory(models.Model):
    """
    Model to store prediction records and fact-checking evidence
    for news articles analyzed by users.
    """
    VERDICT_CHOICES = [
        ('LIKELY_REAL', 'Likely Real'),
        ('POTENTIALLY_FAKE', 'Potentially Fake'),
        ('UNCERTAIN', 'Uncertain – Further verification required'),
    ]

    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='predictions',
        null=True, 
        blank=True
    )
    news_title = models.CharField(max_length=350, blank=True, help_text="Extracted headline or title")
    news_text = models.TextField(help_text="Full news headline or article text analyzed")
    
    # Machine Learning Outputs
    ml_prediction = models.CharField(max_length=50, help_text="Predicted label: REAL, FAKE, or UNCERTAIN")
    ml_confidence = models.FloatField(help_text="Model confidence score percentage (0-100)")
    real_probability = models.FloatField(default=0.0, help_text="Calibrated probability for Real class")
    fake_probability = models.FloatField(default=0.0, help_text="Calibrated probability for Fake class")
    
    # Evidence / Retrieval Outputs
    evidence_status = models.CharField(
        max_length=50, 
        default='UNAVAILABLE',
        help_text="SUPPORTING, CONFLICTING, INSUFFICIENT, or UNAVAILABLE"
    )
    evidence_summary = models.TextField(blank=True, help_text="Synthesis of external evidence")
    retrieved_sources = models.JSONField(default=list, blank=True, help_text="List of source articles retrieved")
    # Final Unified Status & Fact-Check Status
    fact_check_status = models.CharField(max_length=50, default='Not Available', blank=True, help_text="Verified Fact Check result")
    is_current_news = models.BooleanField(default=False, help_text="Whether the claim concerns breaking/current events")
    final_verdict = models.CharField(max_length=120, help_text="Final user-facing classification status")
    badge_class = models.CharField(max_length=20, default='secondary', blank=True)
    explanation = models.TextField(blank=True, help_text="Plain-English explanation of why this verdict was generated")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Prediction History"
        verbose_name_plural = "Prediction Histories"

    def __str__(self):
        snippet = (self.news_text[:50] + '...') if len(self.news_text) > 50 else self.news_text
        return f"[{self.ml_prediction} ({self.ml_confidence:.1f}%)] {snippet}"
