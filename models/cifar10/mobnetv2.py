import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces BatchNorm2d with GroupNorm.
    """
    # Iterate over immediate children modules
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Determine number of channels
            num_channels = child.num_features
            
            # Ensure num_groups doesn't exceed num_channels
            groups = num_groups
            if num_channels < groups:
                # Fallback: if channels are fewer than default groups (e.g. 16 or 24 channels)
                # We make groups smaller to fit.
                groups = int(num_channels / 2) if num_channels > 1 else 1
            elif num_channels % groups != 0:
                # Fallback if channels aren't divisible by groups
                groups = num_channels // 2 

            # Create GroupNorm layer
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels)
            
            # Optional: Copy weights (gamma) and bias (beta) from BN to GN
            gn.weight = child.weight
            gn.bias = child.bias
            
            # Replace the layer in the parent module
            setattr(module, name, gn)
        else:
            # Recurse deeper
            replace_bn_with_gn(child, num_groups)

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device, use_imagenet=False):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.device = device

        # 1. Load MobileNetV2 with ImageNet weights
        if use_imagenet:
            self.raw_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            self.raw_model = mobilenet_v2(weights=None)

        # 2. Adjust for CIFAR-10 (Smaller images)
        # MobileNetV2 structure: features[0] is the first ConvBNReLU block. 
        # features[0][0] is the specific Conv2d layer.
        # Original: stride=2 (for 224x224). We change to stride=1 for 32x32 images.
        self.raw_model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # 3. Convert BatchNorm to GroupNorm
        # Note: MobileNetV2 has many layers with low channel counts (e.g., 16, 24).
        # The replace function handles the case where num_groups=32 > num_channels.
        replace_bn_with_gn(self.raw_model, num_groups=32)

        # 4. Extract the feature extractor
        # In MobileNetV2, 'features' contains all convolution layers.
        # Unlike ResNet, it does not contain the pooling layer, so we don't need to slice children.
        self.feature_extractor = self.raw_model.features

        # 5. New classifier for our num_classes
        # MobileNetV2 output channels are typically 1280.
        # We must add the Pooling layer here because 'self.features' does not include it.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, self.num_classes) 
        )

        # 6. Cleanup: delete the original classifier to save memory/confusion
        del self.raw_model

        self.size = self.model_size()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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