from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render, redirect
import torch
import cv2
import numpy as np
from torchvision import transforms
from django.core.files.storage import FileSystemStorage
from .forms import UploadedFileForm
from .models import UploadedFile
from PIL import Image
from django.conf import settings
import threading
import os
from .cnn import SignatureCNN


# Load the trained model
model = SignatureCNN()
model_path = os.path.join(settings.BASE_DIR, 'signature_detection', 'models', 'final_signature_detection.pth')
model.load_state_dict(torch.load(model_path))
model.eval()

# Define preprocessing transformations
live_feed_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(),

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define preprocessing transformations for uploaded images
upload_image_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match the input size
    transforms.Grayscale(),           # Convert to grayscale
    transforms.ToTensor(),            # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])


# Function to perform inference
def predict_signature(image_path):
    # Open the image file
    image = Image.open(image_path)  # Convert to grayscale
    # Apply the same transformations used in the live feed
    processed_image = upload_image_transform(image)  # Transform should work directly on the PIL image
    processed_image = processed_image.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    output = model(processed_image)
    _, predicted = torch.max(output.data, 1)
    classes = ['forged', 'genuine']
    return classes[predicted.item()]



import cv2
import torch
import numpy as np
from django.http import StreamingHttpResponse

# Assuming you have a signature detection model loaded as 'model'
# and a transformation function 'live_feed_transform' defined elsewhere

def detect_signature(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to detect signature presence
    _, thresh = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return True if contours are found, indicating the presence of a signature
    return len(contours) > 0

def gen(camera):
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect if there is any signature in the frame
        if detect_signature(frame):
            # Assume the signature is a region of interest (ROI) in the center of the frame
            height, width = gray_frame.shape
            roi = gray_frame[int(height / 4):int(3 * height / 4), int(width / 4):int(3 * width / 4)]

            # Check if the ROI contains a signature based on pixel intensity
            mean_intensity = np.mean(roi)
            intensity_threshold = 100  # Adjust this based on your tests

            if mean_intensity < intensity_threshold:
                # Resize the ROI to match the input size of the model
                resized_roi = cv2.resize(roi, (128, 128))
                
                # Preprocess the image
                processed_image = live_feed_transform(resized_roi)

                # Add a batch dimension (batch size of 1)
                processed_image = processed_image.unsqueeze(0)

                # Perform inference using your signature classification model
                output = model(processed_image)
                _, predicted = torch.max(output.data, 1)
                classes = ['forged', 'genuine']
                prediction = classes[predicted.item()]

                # Display the prediction on the frame
                cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            # Display "No signature detected" if no signature is found
            cv2.putText(frame, "No signature detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def live_feed(request):
    return StreamingHttpResponse(gen(cv2.VideoCapture(0)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')



def upload_file(request):
    prediction = None
    uploaded_image = None 

    if request.method == 'POST':
        form = UploadedFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            uploaded_image = uploaded_file.image_file.url

            # Predict the Signature using uploaded file
            prediction = predict_signature(uploaded_file.image_file.path)
            
            return render(request, 'signature_detection/upload.html', {
                'form': form,
                'prediction': prediction,
                
                "uploaded_image": uploaded_image,
            })
    else:
        form = UploadedFileForm()

    return render(request, 'signature_detection/upload.html', {
        'form': form,
        "prediction": prediction,
        'uploaded_image': uploaded_image
        
        })

    


def index(request):
    return render(request, 'signature_detection/index.html')