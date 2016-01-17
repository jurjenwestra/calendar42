# calendar42
A simple API Proxy with caching

The assignment
--------------
Here I list a few thoughts and considerations.


Problem analysis
----------------
Things such as serving http responses and url mapping
should be provided by a web framework (Django). Python
has the json standard library for encoding/decoding
json, but since this is a web standard the framework
probably has support for json.

More interesting is probably the cache implementation.
Caches are used to prevent expensive function calls,
probably more typically in terms of cpu cycles. We want
to write reusable software, and I list a number of
considerations here.


Q: What should get cached?

The assignment states that the results should be cached
for 4.2 minutes. However, we can cache either the results
of our GET endpoint, or the results of the calls to the
c42 API (or both of course). The former seems what is
suggested and at first thought is the most sensible, but
in a more realistic situation there may be other uses for
the data provided by the c42 API. Therefore I have decided
to cache the data returned by the c42 API instead.

In the case at hand, not all data that is returned by the
c42 API is actually used. Storing it can therefore be
considered wasteful. However, since there may be other
uses for the data we collect we store the full original
data. Since the API provides json we store json.


Q: Where should the cached data reside?

Conventionally data is stored in-memory, but it is
certainly possible to databases or flat files instead. I
will use in-memory caching since I have no reason to
choose something else. If possible however we should keep
the other options open.


Q: How do we limit cache size?

Unless we perform some kind of clean up caches may grow
without limits. The assignment mentions a cache time of
4.2 minutes, and it makes sense to expunge expired data.
However, a natural implementation only cleans up on
data access. Our cache will limit the amount of data that
is stored and provides a hook for a background service
that can e.g. at a certain interval perform cleanup.



Code organization

Based on the assignment I have chosen Django as the web
framework. I have tried to follow Django conventions
for the organization of the code. I have created a c42
library that provides the cached API calls. A separate 
cachelib library provides the cache.

I have used the requests library to access the c42 API.
This seems a standard in the python world (it is even
mentioned in its documentation).


How to run
----------
I have used python 2.7.11 and Django 1.9.1 with a virtual env.

1. env\Scripts\activate
2. cd mysite
3. python manage.py runserver
4. Go to http://127.0.0.1:8000/events-with-subscriptions/527c1675e415c4f0becd7a7441e66b6d_14507820169300/
   in your browser to fetch the json
