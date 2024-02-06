from PIL import Image
import os, os.path
from transformers import CLIPModel, CLIPProcessor
import torch
import uuid
import pickle
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
import random
import glob


def load_model_and_processor() -> [CLIPModel, CLIPProcessor]:
    """
    Load model and processor for the Clip model
    :return: void
    """

    # Define the model ID
    model_id = os.environ["MODEL_ID"]
    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # initialize the model for embeddings
    # Save the model to device
    model = CLIPModel.from_pretrained(model_id).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_id)
    # Return model, processor
    return model, processor


def embed_image(processor: CLIPProcessor, model: CLIPModel, image: Image) -> list:
    """
    embed one image int vector using CLIP
    :param processor: clip processor
    :param model: clip model to embed the image
    :param image: object image to be embedded by the model
    :return:
    """
    image = processor(text=None, images=image, return_tensors="pt")["pixel_values"]
    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array
    embedding_as_np = embedding.cpu().detach().numpy()
    return embedding_as_np.tolist()[0]


def process_and_ingest_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    client: QdrantClient,
    collection_name: str,
) -> str:
    """

    :param model: clip model loaded
    :param processor: clip processor
    :param client: Qdrant client
    :param collection_name: name of the qdrant vector collection
    :return: path for the pickle which include all the processed images
    """
    dir_name = os.environ["IMAGES_DIR"]
    batch_size = int(os.environ["BATCH_SIZE"])
    data_limit = int(os.environ["DATA_LIMIT"])
    print(f"processing images in {batch_size} batch size...")
    path = f"data/{dir_name}"
    processed_files_path = f"data/{dir_name}.pkl"
    pkl_file = open(processed_files_path, "wb")
    valid_images = [".jpg", ".gif", ".png"]
    images_paths = [
        file
        for file in glob.glob(path + "/**/*.*", recursive=True)
        if os.path.splitext(file)[1] in valid_images
    ]
    if data_limit:
        images_paths = random.choices(images_paths, k=data_limit)
    images_batches = cut_to_batches(images_paths, batch_size)
    for batch in images_batches:
        images_embeddings_batch = []
        for file in tqdm(batch, position=0, leave=True):
            image_file = Image.open(file).convert("RGB")
            images_payload = {
                "id": str(uuid.uuid4()),
                "payload_info": {"uri": f"{file}"},
                "embedding": embed_image(processor, model, image_file),
            }
            images_embeddings_batch.append(images_payload)
        ingest_to_qdrant(client, collection_name, images_embeddings_batch)
        pickle.dump(images_embeddings_batch, pkl_file)
    return processed_files_path


def cut_to_batches(my_list: list, batch_size: int):
    """
    Cut a list of objects into batches
    :param my_list: list of objects
    :param batch_size: size of the batch to cut the list into
    :return:
    """
    for i in range(0, len(my_list), batch_size):
        yield my_list[i : i + batch_size]


def ingest_to_qdrant(client, collection_name, batch):
    """
    ingest batch of images vectors into Qdrant
    :param client: client to connect to qdrant
    :param collection_name: name of the collection to which the images will be included
    :param batch: batch of images vectors
    :return:
    """
    image_vectors = [image["embedding"] for image in batch]
    payloads = [image["payload_info"] for image in batch]
    ids = [image["id"] for image in batch]
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(vectors=image_vectors, payloads=payloads, ids=ids),
    )
    print("Batch ingested to Qdrant ...")


def load_processed_files(processed_files_path):
    with open(processed_files_path, "rb") as file_handle:
        images_info = pickle.load(file_handle)
    return images_info
