from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('chat/', views.chat, name='chat'),
    path('chat/<int:session_id>/', views.chat, name='chat_session'),
    path('start_new_session/', views.start_new_session, name='start_new_session'),
    path('delete_session/<int:session_id>/', views.delete_session, name='delete_session'),
    path('rename_session/<int:session_id>/', views.rename_session, name='rename_session'),
    path('api/chat/<int:session_id>/', views.ChatAPI.as_view(), name='chat_api'),
]