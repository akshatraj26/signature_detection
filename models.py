# Model Architecture
from torch import nn
from torch.nn import functional as F

class SignatureCNN(nn.Module):
    def __init__(self):
        super(SignatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        # we have 2 classes genuine and forge
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        # Flatten the Tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x