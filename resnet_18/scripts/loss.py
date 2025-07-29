import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio

class FocalLoss(nn.Module):
    """
    Implements Focal Loss 
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            raise TypeError("alpha must be a Tensor or None.")
        # Register alpha as a buffer if it's a tensor
        if isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = alpha # Will be None

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate Cross Entropy loss without reduction to get per-example loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate pt (probability of the true class) ... pt = exp(-ce_loss) since ce_loss = -log(pt) for the correct class
        pt = torch.exp(-ce_loss)

        # Calculate the focal loss (1 - pt)^gamma
        focal_term = (1 - pt)**self.gamma

        # Calculate the final focal loss: alpha_t * (1 - pt)^gamma * ce_loss
        loss = focal_term * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # else: ('none') return the per-example loss tensor

        return loss



# class CustomNormalization(tio.Lambda):
#     def __init__(self, max_value, mean, std):
#         self.max_value = max_value
#         self.mean = mean
#         self.std = std
#         super().__init__(self.apply)

#     def apply(self, subject):
#         # Normalize each volume in the subject
#         for name, image in subject.items():
#             image.data = (image.data / self.max_value - self.mean) / self.std
#         return subject

# class ResidualBlock3D(nn.Module):
#     """Basic 3D residual block with skip connections."""
#     def __init__(self, in_channels, out_channels, downsample=False):
#         super(ResidualBlock3D, self).__init__()

#         stride = (1,2,2) if downsample else (1,1,1)  # Avoid early Z-downsampling

#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3,5,5), stride=stride, padding=(1,2,2))
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3,5,5), padding=(1,2,2))
#         self.bn2 = nn.BatchNorm3d(out_channels)

#         if downsample or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
#                 nn.BatchNorm3d(out_channels)
#             )
#         else:
#             self.shortcut = nn.Identity()

#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += residual 
#         return F.relu(x)  # Final activation

# class NeurotransmitterResNet3D(nn.Module):
#     def __init__(self, in_channels=1, num_classes=5):
#         super(NeurotransmitterResNet3D, self).__init__()

#         self.initial_conv = nn.Conv3d(in_channels, 32, kernel_size=(3,5,5), padding=(1,2,2))

#         self.block1 = ResidualBlock3D(32, 64, downsample=True)   # (14, 80, 80)
#         self.block2 = ResidualBlock3D(64, 128, downsample=True)  # (14, 40, 40)
#         self.block3 = ResidualBlock3D(128, 256, downsample=True) # (14, 20, 20)

#         # Downsample depth only once at later stage
#         self.depth_pool = nn.MaxPool3d(kernel_size=(2,1,1))  # (7, 20, 20)

#         self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Reduce to 1x1x1
#         self.dropout = nn.Dropout(p=0.2)
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = x.permute(0,1,4,2,3)
#         x = F.relu(self.initial_conv(x))
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)

#         x = self.depth_pool(x)
#         x = self.global_avg_pool(x).view(x.shape[0], -1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x