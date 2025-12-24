from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from django.conf import settings
import json
import requests
import base64
import io
from .models import ChatSession, Message
from .assistant import DrPatoAssistant

assistant = DrPatoAssistant()

def home(request):
    if request.user.is_authenticated:
        sessions = ChatSession.objects.filter(user=request.user)
        if sessions.exists():
            return redirect('chat_session', session_id=sessions.first().id)
        else:
            return redirect('start_new_session')
    else:
        return redirect('login')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'core/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('chat')
    return render(request, 'core/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def chat(request, session_id=None):
    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        messages = Message.objects.filter(session=session).order_by('timestamp')
    else:
        session = None
        messages = []
    sessions = ChatSession.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'core/chat.html', {'sessions': sessions, 'current_session': session, 'messages': messages})

@login_required
def start_new_session(request):
    session = ChatSession.objects.create(user=request.user, title="New Chat")
    return redirect('chat_session', session_id=session.id)

@login_required
def delete_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.delete()
    return redirect('chat')

@login_required
def rename_session(request, session_id):
    if request.method == 'POST':
        import json
        data = json.loads(request.body)
        new_title = data.get('title')
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        session.title = new_title
        session.save()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})

@method_decorator(csrf_exempt, name='dispatch')
class ChatAPI(View):
    def post(self, request, session_id):
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        
        user_message = request.POST.get('message', '')
        image_file = request.FILES.get('image')
        
        # If image uploaded, detect disease first
        disease_result = None
        if image_file:
            try:
                # Convert image to base64 as per requirement
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                print(f"Sending base64 image, length: {len(image_base64)}")
                
                # Call FastAPI with base64
                payload = {"image": image_base64}
                print(f"Payload keys: {list(payload.keys())}")
                response = requests.post('http://127.0.0.1:8001/predict_base64', json=payload)
                print(f"FastAPI response status: {response.status_code}")
                print(f"FastAPI response: {response.text}")
                if response.status_code == 200:
                    disease_result = response.json().get('disease')
                    print(f"Detected disease: {disease_result}")
                else:
                    disease_result = "Error detecting disease"
            except Exception as e:
                print(f"Error calling FastAPI: {e}")
                disease_result = f"Error: {str(e)}"
        
        # Create user message
        message = Message.objects.create(session=session, role='user', content=user_message)
        if image_file:
            # Reset file pointer for saving
            image_file.seek(0)
            message.image = image_file
            message.save()
        
        # Create assistant instance
        assistant = DrPatoAssistant()
        # Load history
        messages = Message.objects.filter(session=session).order_by('timestamp')
        assistant.conversation_history = [{"role": "system", "content": assistant.system_prompt}]
        for msg in messages:
            assistant.conversation_history.append({"role": msg.role, "content": msg.content})
        
        # If disease detected from image, handle it
        if disease_result:
            response_content = assistant.handle_disease_detection(disease_result)
        else:
            response_content = assistant.chat(user_message)
        
        Message.objects.create(session=session, role='assistant', content=response_content)
        
        # Auto-rename if this is the first assistant message
        if Message.objects.filter(session=session, role='assistant').count() == 1:
            session.title = ' '.join(response_content.split()[:5])
            session.save()
        return JsonResponse({'response': response_content})
        # Auto-rename if this is the first assistant message
        if Message.objects.filter(session=session, role='assistant').count() == 1:
            session.title = ' '.join(response.split()[:5])
            session.save()
        return JsonResponse({'response': response})
