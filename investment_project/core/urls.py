from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('assistant/', views.home, name='home'),
    path('predict', views.predict, name='predict'),
]