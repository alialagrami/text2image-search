from qdrant_client import QdrantClient
import os


class QdrantInstance:
    def __init__(self):
        self.host = None
        self.port = None
        self.qdrant_client = None

    async def connect(self):
        self.host = os.environ["HOST"]
        self.port = int(os.environ["PORT"])
        self.qdrant_client = QdrantClient(host=self.host, port=self.port)
