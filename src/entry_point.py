# References
# https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html

import logging

import numpy as np
import PIL
from numpy import ndarray as NDArray
from PIL.Image import Image
from six import BytesIO
from torch.nn import Module

from facenet_pytorch import MTCNN


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir: str) -> Module:

    return MTCNN(image_size=160, margin=0, device="cuda:0")


def input_fn(request_body: bytes, content_type: str = "application/x-npy") -> Image:

    stream = BytesIO(request_body)
    np_img = np.load(stream, allow_pickle=True)
    return PIL.Image.fromarray(np_img)


def predict_fn(input_data: Image, model: Module) -> NDArray:

    face = model(input_data)
    face = face.permute(1, 2, 0)
    return face.detach().cpu().numpy()


def output_fn(prediction: NDArray, content_type: str = "application/x-npy") -> bytes:

    buffer = BytesIO()
    np.save(buffer, prediction)
    return buffer.getvalue()
