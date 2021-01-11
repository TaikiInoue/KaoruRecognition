# References
# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html

import logging
import os

import numpy as np
import PIL
import torch
from numpy import ndarray as NDArray
from six import BytesIO
from torch import Tensor
from torch.nn import Module
from torchvision import transforms
from torchvision.models import resnet50


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir: str) -> Module:

    logger.info("START: model_fn")
    model = resnet50(pretrained=False)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    logger.info("END: model_fn")
    return model


def input_fn(request_body: bytes, content_type: str = "application/x-npy") -> Tensor:

    logger.info("START: input_fn")
    stream = BytesIO(request_body)
    np_img = np.load(stream, allow_pickle=True)
    pil_img = PIL.Image.fromarray(np_img)
    preprocess = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor_img = preprocess(pil_img)
    tensor_img = tensor_img.unsqueeze(0)
    logger.info("END: input_fn")
    return tensor_img


def predict_fn(input_data: Tensor, model: Module) -> NDArray:

    logger.info("START: predict_fn")
    model.to("cuda:0")
    model.eval()
    with torch.no_grad():
        prediction = model(input_data.to("cuda:0"))
        logger.info("END: predict_fn")
        return prediction.detach().cpu().numpy()


def output_fn(prediction: NDArray, content_type: str = "application/x-npy") -> bytes:

    logger.info("START: output_fn")
    buffer = BytesIO()
    np.save(buffer, prediction)
    logger.info("END: output_fn")
    return buffer.getvalue()
