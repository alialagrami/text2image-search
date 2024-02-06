import os

from utils.helpers import process_and_ingest_images, load_model_and_processor, ingest_to_qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
import yaml


def process_and_ingest_images_into_qdrant(collection_name: str):
    """
    :param collection_name: name of the collection to be created
    :return: void
    """
    client = QdrantClient(host=os.environ["HOST"], port=int(os.environ["PORT"]))
    try:
        client.get_collection(collection_name=collection_name)
    except Exception as error:
        print("Creating new collection")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=os.environ["VECTOR_SIZE"], distance=models.Distance.COSINE)
        )

    model, processor = load_model_and_processor()

    process_and_ingest_images(
        model=model,
        processor=processor,
        client=client,
        collection_name=collection_name)


if __name__ == "__main__":
    def pes(*args, **kw):
        raise NotImplementedError
    with open("config.yml", "r") as stream:
        try:
            yaml.parser.Parser.process_empty_scalar = pes
            CONFIG = yaml.safe_load(stream)
            print(CONFIG)
        except yaml.YAMLError as exc:
            print(exc)

    os.environ["HOST"] = CONFIG["QDRANT_CLIENT_CONFIG"]["HOST"]
    os.environ["PORT"] = str(CONFIG["QDRANT_CLIENT_CONFIG"]["PORT"])
    os.environ["IMAGES_DIR"] = CONFIG["DATA_INGESTION_CONFIG"]["IMAGES_DIR"]
    os.environ["BATCH_SIZE"] = str(CONFIG["DATA_INGESTION_CONFIG"]["BATCH_SIZE"])
    os.environ["DATA_LIMIT"] = str(CONFIG["DATA_INGESTION_CONFIG"]["DATA_LIMIT"])
    os.environ["MODEL_ID"] = CONFIG["MODEL_CONFIG"]["MODEL_ID"]
    os.environ["VECTOR_SIZE"] = str(CONFIG["MODEL_CONFIG"]["VECTOR_SIZE"])
    process_and_ingest_images_into_qdrant(
        collection_name="images_search"
    )


