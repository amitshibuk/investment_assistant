from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('landing/', views.landing, name='landing'),
    path('assistant/', views.home, name='home'),
    path('predict', views.predict, name='predict'),
    
    # Portfolio
    path('portfolio/', views.portfolio, name='portfolio'),
    path('add-portfolio/', views.add_to_portfolio, name='add_to_portfolio'),
    path('remove-portfolio/', views.remove_from_portfolio, name='remove_from_portfolio'),
    path('api/portfolio-data/<str:ticker>/', views.get_portfolio_data, name='get_portfolio_data'),
    path('api/portfolio-detail/<str:ticker>/', views.get_portfolio_detail, name='get_portfolio_detail'),

    # Market Data
    path('api/market-status/', views.get_market_status, name='get_market_status'),
    path('api/stock-news/<str:ticker>/', views.get_stock_news, name='get_stock_news'),
    path('api/top-stocks/', views.get_top_stocks, name='get_top_stocks'),
    path('api/live-tickers/', views.get_live_tickers, name='get_live_tickers'),

    # Auto-Train
    path('api/auto-train/<str:ticker>/', views.auto_train, name='auto_train'),
]