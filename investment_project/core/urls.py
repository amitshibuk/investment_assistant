from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'), # New Home
    path('landing/', views.landing, name='landing'), # Legacy Landing
    path('assistant/', views.home, name='home'),     # Legacy Chat
    path('predict', views.predict, name='predict'),
    
    # Portfolio
    path('portfolio/', views.portfolio, name='portfolio'),
    path('add-portfolio/', views.add_to_portfolio, name='add_to_portfolio'),
    path('api/portfolio-data/<str:ticker>/', views.get_portfolio_data, name='get_portfolio_data'),
]