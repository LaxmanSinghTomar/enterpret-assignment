
import sys
sys.path.insert(1, "app/")
sys.path.insert(2, "src/")
import fastapi
from fastapi import FastAPI, Request, status
from pydantic import BaseModel
from src.inference import summarize
import json
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI(title="Summarisation App", description="Summarisation App", version=0.1)

class SummaryInput(BaseModel):
    product_name: str
    record_type: str
    text: str


@app.get("/", tags=["Health Check"])
def healthcheck(request: Request):
    """
    Health Check.

    Args:
        request (request): Request Parameter

    Returns:
        response (dict): Response
    """
    response_body = {"status": "Service is Healthy & Running!"}
    response = JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(response_body))
    return response


@app.post("/summarise")
async def summarise(summary_input: SummaryInput):
    input_json = json.loads(summary_input.json())
    try:
        output = summarize(text=input_json['text'], record_type=input_json['record_type'])
        resp_status="success!"
    except Exception as e:
        print(e)
        resp_status="failure!"
    if resp_status == "success!":
        response_body = {"status": resp_status, "result": output}
        response = JSONResponse(
            status_code=fastapi.status.HTTP_200_OK,
            content=jsonable_encoder(response_body),
        )
    else:
        response_body = {
            "status": "Task Failed!",
        }
        response = JSONResponse(
            status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=jsonable_encoder(response_body),
        )
    return response