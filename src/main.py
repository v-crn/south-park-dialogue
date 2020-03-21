import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
import os
from typing import List

import sys
sys.path.append('src')
import config as c
import utils as u
from preprocessing import preprocess
from modeling import build, train_and_validate, train_predict_cv


# init app
app = FastAPI()

global df
if os.path.exists(c.PATH_TRAIN_EXTRA):
    df = u.load(c.PATH_TRAIN_EXTRA)
else:
    df = pd.read_csv(c.PATH_TRAIN)

# Routes
@app.get('/')
async def index():
    return {"text": "Hello API Builders"}


@app.post("/line/")
async def add_line(line: c.Line):
    print(line)
    extra_lines = pd.DataFrame([line.dict()])

    global df
    print(df.shape)
    df = pd.concat([df, extra_lines], axis=0, ignore_index=True)
    print(df.shape)
    u.dump(df, c.PATH_TRAIN_EXTRA)
    return line


@app.post("/lines/")
async def add_lines(lines: List[c.Line]):
    print(lines)
    extra_lines = []
    for line in lines:
        extra_lines.append(line.dict())

    global df
    print(df.shape)
    df = pd.concat([df, pd.DataFrame(extra_lines)], axis=0, ignore_index=True)
    print(df.shape)
    u.dump(df, c.PATH_TRAIN_EXTRA)
    return line


@app.get("/lines/")
async def read_lines():
    print(df.shape)
    return str(df)


@app.get("/model/{model_name}")
async def get_model(model_name: c.ModelName):
    if model_name == c.ModelName.lgbm:
        return {"model_name": model_name, "message": "LightGBM"}

    return {"model_name": model_name, "message": "LightGBM"}


@app.get('/train/')
async def train():
    print(df.shape)
    X, y = preprocess(df)
    model = build(c.ModelName.lgbm)
    model = train_and_validate(model, X, y)
    return "Training is done."


@app.get('/evaluate/')
async def evaluate():
    X, y = preprocess(df)
    alg_compare, feature_importances_compare\
        = train_predict_cv(c.MLA, X, y)
    alg_compare.to_csv(c.PATH_MLA_COMPARE)
    feature_importances_compare.to_csv(c.PATH_MLA_FEATURE_IMPORTANCES)
    return str(alg_compare)


@app.get('/predict/')
async def predict(text: str = Query(None, min_length=2, max_lengh=407)):
    # Vectorizer
    vectorizer = u.load(c.PATH_VECTORIZER)

    # Label Encoder
    le = u.load(c.PATH_ENCODER)

    # Models
    model = u.load(c.PATH_MODEL)

    text_array = np.array([text])
    X = vectorizer.transform(text_array).toarray()

    prediction = model.predict(X)
    name = le.inverse_transform(prediction)[0]

    return {"Text": text, "Character": name}


@app.post('/predict/{text}')
async def predict(text):
    # Vectorizer
    vectorizer = u.load(c.PATH_VECTORIZER)

    # Label Encoder
    le = u.load(c.PATH_ENCODER)

    # Models
    model = u.load(c.PATH_MODEL)

    text_array = np.array([text])
    X = vectorizer.transform(text_array).toarray()

    prediction = model.predict(X)
    name = le.inverse_transform(prediction)[0]

    return {"Text": text, "Character": name}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
