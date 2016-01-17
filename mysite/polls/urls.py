from django.conf.urls import url

from . import views

urlpatterns = [
  url(r'^([0-9]+)/$', views.jwestra_index, name='jwestra_index'),
  url(r'^$', views.index, name='index'),
]