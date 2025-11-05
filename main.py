import torch, random
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

from transformer_model import CzecherTransformer
from czecher_tokenizer.bpe_tokenizer import GPTTokenizer
from datasets import load_dataset

torch.set_grad_enabled(False)


def get_params(layers: int = 12):
    model_dim = layers * 64
    return model_dim * 64, max(1, (model_dim + 127) // 128), 4 * model_dim


class PunctuateRequest(BaseModel):
    text: str = Field(..., description="Raw, unpunctuated text")
    threshold: Optional[float] = Field(
        0.5, ge=0.0, le=1.0, description="Confidence threshold (0-1)"
    )


class PunctuateResponse(BaseModel):
    punctuated: str
    seconds: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f'Loading dataset...')
    dataset = load_dataset("josefbednar/11m-czech-sentences", split="train")

    print("Initializing tokenizer...")
    tokenizer = GPTTokenizer(json_file='syn2006pub_11m_tokenizer.json')
    print("Initializing model...")
    model = CzecherTransformer.load('11m_4xGPU_12layers.pt', map_location='cpu')
    model.eval()

    app.state.dataset = dataset
    app.state.model = model
    app.state.tokenizer = tokenizer
    print("Ready for production.")
    yield

app = FastAPI(title="CzecherAPI", version="1.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/random-text")
def random_text(request: Request):
    dataset = request.app.state.dataset
    return {"text": dataset[random.randint(0, len(dataset) - 1)]['text']}

@app.post("/punctuate", response_model=PunctuateResponse)
def punctuate_endpoint(payload: PunctuateRequest, request: Request):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty.")
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    try:
        with torch.inference_mode():
            seconds, punctuated = model.punctuate(
                text=text, tokenizer=tokenizer,
                threshold=payload.threshold or 0.5, device="cpu",
            )
        return PunctuateResponse(punctuated=punctuated, seconds=seconds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Punctuation failed: {e}")
