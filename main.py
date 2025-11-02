from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
import torch
import torch.nn as nn
from contextlib import asynccontextmanager

from transformer_model import CzecherTransformer
from czecher_tokenizer.bpe_tokenizer import GPTTokenizer

torch.set_grad_enabled(False)


def get_params(layers: int = 12):
    model_dim = layers * 64
    return model_dim * 64, max(1, (model_dim + 127) // 128), 4 * model_dim


# num_layers = 12
# model_dim, num_heads, dim_ff = get_params(layers=num_layers)

# print(f'Initializing model and tokenizer...')
# tokenizer = GPTTokenizer(json_file='tokenizer.json')
# print(f'Tokenizer loaded.')
# model = CzecherTransformer(vocab_size=tokenizer.get_vocab_size(), pad_id=tokenizer.get_pad_token_id(), max_tokens=128, num_layers=num_layers, d_model=model_dim, embedding_dim=model_dim, nhead=num_heads, dim_ff=dim_ff)
# model = model.load('10m_4xGPU_12layer_2epoch_1.pt', map_location='cpu')
# model.eval()
# print(f'Model and tokenizer successfully loaded, ready for requests.')

# app = FastAPI(title="CzecherAPI", version="1.0.0")


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
    print("Initializing model and tokenizer...")
    tokenizer = GPTTokenizer(json_file='tokenizer.json')
    print(f'Tokenizer loaded.')
    model = CzecherTransformer.load('10m_4xGPU_12layer_2epoch_1.pt', map_location='cpu')
    model.eval()

    app.state.model = model
    app.state.tokenizer = tokenizer
    print("Model and tokenizer ready.")
    yield

app = FastAPI(title="CzecherAPI", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

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
    