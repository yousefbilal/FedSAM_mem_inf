import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces BatchNorm2d with GroupNorm.
    """
    # Iterate over immediate children modules
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Determine number of channels
            num_channels = child.num_features
            
            # Ensure num_groups doesn't exceed num_channels (rare edge case)
            groups = num_groups
            if num_channels % groups != 0:
                # Fallback if channels aren't divisible by groups
                # For ResNet (64, 128, 256, 512), 32 is always safe.
                groups = num_channels // 2 

            # Create GroupNorm layer
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels)
            
            # Optional: Copy weights (gamma) and bias (beta) from BN to GN
            # This helps preserve some learned scaling if using pretrained weights
            gn.weight = child.weight
            gn.bias = child.bias
            
            # Replace the layer in the parent module
            setattr(module, name, gn)
        else:
            # Recurse deeper
            replace_bn_with_gn(child, num_groups)

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.device = device

        # 1. Load ResNet18 with ImageNet weights
        # We use this directly instead of creating a separate 'm' variable
        self.feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 2. Adjust for CIFAR-10 (Smaller images)
        # Note: This resets the weights for this specific layer
        self.feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature_extractor.maxpool = nn.Identity()

        # 3. Convert BatchNorm to GroupNorm BEFORE splitting the model
        # We do this now while the model structure is still standard ResNet
        replace_bn_with_gn(self.feature_extractor, num_groups=32)

        # 4. Remove the last layer (original fc)
        # We wrap in Sequential to make it a feature extractor
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

        # 5. New classifier for our num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.num_classes) 
        )

        # 6. Move to device immediately (optional, but good practice)

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