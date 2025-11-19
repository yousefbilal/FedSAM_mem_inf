import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights


class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.device = device

        # Load ResNet18 with random initialization
        self.feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        m = resnet18(weights=None)

        # Adjust for CIFAR-10
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        
        # Remove the last layer (original classifier)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # New classifier for our num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.num_classes)  # ResNet18's last conv layer outputs 512 channels
        )

        # # Initialize weights
        # self._initialize_weights()

        self.size = self.model_size()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size