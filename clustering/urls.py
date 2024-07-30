from django.urls import path
from .views import categorize_questions_view

urlpatterns = [
    path('categorize/', categorize_questions_view, name='categorize_questions'),
]
