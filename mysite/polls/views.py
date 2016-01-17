from django.http import HttpResponse

def index(request):
  return HttpResponse("Hello, world. You're at the polls index.")

def jwestra_index(request, arg):
  return HttpResponse("Your argument was '%s'" % (arg))
 