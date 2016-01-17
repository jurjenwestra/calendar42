from django.conf.urls import url

from . import views

urlpatterns = [
  url(r'^(\S+)/$', views.event, name='event'),
  url(r'^$', views.index, name='index'),
]