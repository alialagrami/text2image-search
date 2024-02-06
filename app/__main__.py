import os
from fastapi import FastAPI, Request

from api import api
from core.qdrant_client.qdrant_session import qdrant_instance
from core.clip_model.clip_session import clip_instance
import uvicorn
import yaml
from pydantic import BaseModel

app = FastAPI()


class SearchInput(BaseModel):
    input_text: str


@app.on_event("startup")
async def startup():
    await qdrant_instance.connect()
    app.state.qdrant = qdrant_instance
    await clip_instance.load()
    app.state.clip = clip_instance


@app.post("/semantic_search/")
async def get_relevant_images(request: Request, search_input: SearchInput):
    return api.get_relevant_images(request, search_input)


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
    os.environ["MODEL_ID"] = CONFIG["MODEL_CONFIG"]["MODEL_ID"]
    uvicorn.run(app, host="0.0.0.0", port=8000)
