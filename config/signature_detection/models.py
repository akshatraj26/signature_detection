from django.db import models
from datetime import datetime
import os 

# Create your models here.
def uploaded_path(inatance, filename):
    return os.path.join('uploads', filename)

class UploadedFile(models.Model):
    image_file = models.FileField(upload_to = uploaded_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image_file.name

