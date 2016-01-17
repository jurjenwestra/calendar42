from collections import OrderedDict

from django.http import HttpResponse, JsonResponse
from django.http import HttpResponseForbidden, HttpResponseNotFound

import c42

import pdb

def index(request):
  return HttpResponse("You're at the index.")
 
def event(request, eventId):
  if request.method != 'GET':
    # Not sure about right response
    return HttpResponseForbidden()
  result = OrderedDict(id=eventId)
  
  try:
    details = c42.get_details_json(eventId)
    subs = c42.get_subscriptions_json(eventId)
  
    # Not sure about the guarantees on the json
    title = details["data"][0]["title"]
    names = [d['subscriber']['first_name'] 
             for d in subs['data']]
  except:
    # Not sure about right response
    return HttpResponseNotFound()
  result["title"] = title
  result["names"] = names
           
  return JsonResponse(result)

