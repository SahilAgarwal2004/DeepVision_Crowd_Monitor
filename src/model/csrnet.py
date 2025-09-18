import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


class CSRNet(nn.Module):
    def __init__(self, pretrained=True):
        super(CSRNet, self).__init__()

        # Load VGG16
        if pretrained:
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg = models.vgg16(weights=None)

        # Layers up to conv4_3 (index 23)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # Backend (dilated convolutions)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)   # Features 
        x = self.backend(x)    # Density map
        return x

    def _initialize_weights(self):
        for module in self.backend.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def get_parameter_groups(self, lr_frontend=1e-6, lr_backend=1e-4):
        return [
            {'params': self.frontend.parameters(), 'lr': lr_frontend},
            {'params': self.backend.parameters(), 'lr': lr_backend}
        ]