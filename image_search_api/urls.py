from django.urls import re_path as url
from .views import FileView

urlpatterns = [
    url(r'^upload/$', FileView.as_view(), name='file-upload'),
]