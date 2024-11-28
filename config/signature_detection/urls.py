from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static


from django.urls import path
from .views import index, live_feed, upload_file

urlpatterns = [
    path("", index, name='index'),
    path("live_feed/", live_feed, name='live_feed'),
    path("upload_file/", upload_file, name='upload_file'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)