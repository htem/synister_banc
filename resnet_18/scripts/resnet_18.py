import torch
import torch.nn as nn
import torchvision.models as models

""" Models used from PyTorch """

def resnet_18_model(num_classes):
    r3d_18_model = models.video.r3d_18(pretrained=False)

    num_input_channels = 1
    original_stem_conv = r3d_18_model.stem[0]

    r3d_18_model.stem[0] = nn.Conv3d(
        in_channels=num_input_channels,
        out_channels=original_stem_conv.out_channels,
        kernel_size=original_stem_conv.kernel_size,
        stride=original_stem_conv.stride,
        padding=original_stem_conv.padding,
        bias=original_stem_conv.bias
    )
    print(f"Modified model's stem input channels to: {num_input_channels}")

    num_classes = 8 # number of neurotransmitter classes
    num_ftrs = r3d_18_model.fc.in_features
    r3d_18_model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Modified model's fc layer output to: {num_classes} classes.")

    return r3d_18_model


def resnet_2_1_model(num_classes):
    r3d_18_model = models.video.r2plus1d_18(pretrained=False)

    num_input_channels = 1
    original_stem_conv = r3d_18_model.stem[0]

    r3d_18_model.stem[0] = nn.Conv3d(
        in_channels=num_input_channels,
        out_channels=original_stem_conv.out_channels,
        kernel_size=original_stem_conv.kernel_size,
        stride=original_stem_conv.stride,
        padding=original_stem_conv.padding,
        bias=original_stem_conv.bias
    )
    print(f"Modified model's stem input channels to: {num_input_channels}")

    num_classes = 8 # number of neurotransmitter classes
    num_ftrs = r3d_18_model.fc.in_features
    r3d_18_model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Modified model's fc layer output to: {num_classes} classes.")

    return r3d_18_model
