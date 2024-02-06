from PIL import Image
import os, os.path
from transformers import CLIPModel, CLIPProcessor
import torch
import numpy as np

class CLIP:
    def __init__(self):
        self._CLIPModel = None
        self._CLIPProcessor = None
        self._device = None

    async def load(self):
        model_id = os.environ["MODEL_ID"]

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # initialize the model for embeddings
        # Save the model to device
        self._CLIPModel = CLIPModel.from_pretrained(model_id).to(self._device)
        # Get the processor
        self._CLIPProcessor = CLIPProcessor.from_pretrained(model_id)

    def get_text_embedding(self, text: str) -> np:
        """
        embed a piece of text
        :param text: text to be embedded
        :return: numpy vector
        """
        image = self._CLIPProcessor(
            text=text,
            return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in image.items()}
        with torch.no_grad():
            text_features = self._CLIPModel.get_text_features(**inputs)
        embedding_as_np = text_features.cpu().detach().numpy()[0]
        return embedding_as_np