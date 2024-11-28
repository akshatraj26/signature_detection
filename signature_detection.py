# Load the dependencies
import numpy
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import SignatureCNN

    

def load_and_preprocess(path):
    # Load the image
    image = Image.open(path)

    # Preprocess the image
    transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    # Apply preprocessing
    image = transform(image)
    # Add a batch dimension
    image = image.unsqueeze(0)
    return image


model = SignatureCNN()
# Load the saved model
model.load_state_dict(torch.load("final_signature_detection.pth"))

model.eval()

# Make predictions
def make_prediction(path):
    image = load_and_preprocess(path)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    
    classes = ['forged', 'genuine']
    return classes[predicted.item()]

# data = ['handwritten signatures/forge2.png', "handwritten signatures/real.png"]

from glob import glob

file = "data/*.png"
data = glob(file)
print(data)

for signature in data:
    Image.open(signature)
    label = make_prediction(signature)
    print(f"Predicted Class:- {label} and original label:- {signature.split("\\")[-1].split(".")[0]}")

