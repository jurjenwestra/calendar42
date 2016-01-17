"""Thin wrapper around c42 API

This module provides access to the c42 API:
  1. it stores / uses the token, and
  2. it caches results
"""

import os

import requests

import cachelib

# Provided by calendar42
TOKEN = '7234acc0bee9f401aa3fcbaf8f99780f987c6542'
EVENT_ID = '527c1675e415c4f0becd7a7441e66b6d_14507820169300'
EXPIRE_SECS = 4.2*60 # Data fetched from c42 'expires' after this
# templates: fill in eventId
DETAILS_URL_T = "https://demo.calendar42.com/api/v2/events/%s/"
SUBSCRIPTIONS_URL_T = "https://demo.calendar42.com/api/v2/event-subscriptions/?event_ids=[%s]"
JSON_HEADERS = {"Accept": "application/json",
                "Content-type": "application/json",
                "Authorization": "Token %s" % (TOKEN)}
# /Provided

#
# C42 api access
#    
@cachelib.ExpiringMemCache.cached(expireSecs=EXPIRE_SECS, 
                                  maxNumKeys=None)
def get_details_json(eventId):
  """Returns detail data for an event as json (or None on error)"""
  URL = DETAILS_URL_T % (eventId)
  r = requests.get(URL,
                   headers=JSON_HEADERS)
  return json_or_None(r)
  
  
@cachelib.ExpiringMemCache.cached(expireSecs=EXPIRE_SECS, 
                                  maxNumKeys=None)
def get_subscriptions_json(eventId):
  """Returns subscription data for an event as json (or None on error)"""
  URL = SUBSCRIPTIONS_URL_T % (eventId)
  r = requests.get(URL,
                   headers=JSON_HEADERS)
  return json_or_None(r)
  
def json_or_None(r):
  """Return None if the status code != ok"""
  return r.json() if r.status_code==requests.codes.ok else None
