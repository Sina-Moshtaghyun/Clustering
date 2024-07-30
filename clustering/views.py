from django.http import JsonResponse
from .utils import categorize_new_questions

def categorize_questions_view(request):
    if request.method == 'POST':
        data = request.json()
        new_questions = data.get('questions', [])
        if not new_questions:
            return JsonResponse({'error': 'No questions provided'}, status=400)
        cluster_counts = categorize_new_questions(new_questions)
        return JsonResponse(cluster_counts)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

