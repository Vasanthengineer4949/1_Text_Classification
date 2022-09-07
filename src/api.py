import config
from transformers import pipeline
from fastapi import FastAPI
import uvicorn

app = FastAPI(debug=True)

@app.get("/")
def home():
    return {"Project Name": "Text Classification"}

@app.get("/predict")
def predict(inp_text:str):
    pipe = pipeline("text-classification", "Vasanth/"+config.MODEL_OUT_NAME)
    return pipe(inp_text)

if __name__ == "__main__":
    uvicorn.run(app)