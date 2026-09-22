from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'ml_prediction', 'ml_confidence', 'evidence_status', 'final_verdict', 'created_at')
    list_filter = ('ml_prediction', 'evidence_status', 'created_at')
    search_fields = ('news_text', 'news_title', 'user__username')
    readonly_fields = ('created_at',)
