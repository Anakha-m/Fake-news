from django.urls import path
from . import views

urlpatterns = [
    # Authentication routes
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),

    # Application routes
    path('', views.dashboard_view, name='dashboard'),
    path('predict/', views.predict_view, name='predict'),
    path('history/', views.history_view, name='history'),
    path('history/<int:pk>/', views.result_detail_view, name='result_detail'),
    path('history/clear/', views.clear_history_view, name='clear_history'),
]
