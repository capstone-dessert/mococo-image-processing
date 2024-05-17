import torch
from torch import Tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes: int):
    # Load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def load_model(model_path, num_classes):
    # Load a model; assuming model architecture is known
    model = get_model_instance_segmentation(num_classes)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def predict(model, preprocessed_image: Tensor, device="cpu"):
    model.eval()
    model.to(device)
    preprocessed_image = preprocessed_image.unsqueeze(0)  # Add batch dimension
    preprocessed_image = preprocessed_image.to(device)
    with torch.no_grad():
        prediction = model(preprocessed_image)
    return prediction
