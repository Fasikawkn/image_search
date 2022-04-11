from django.urls import re_path as url
from .views import FileView, TrainImage


urlpatterns = [
    url(r'^upload/$', FileView.as_view(), name='file-upload'),
    url(r'^train/$', TrainImage.as_view(), name='image-train'),
]