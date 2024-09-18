import torch
from utils.model import Classifier
import json
from utils.scaler import scaler
from fastapi import FastAPI
from pydantic import BaseModel
import time

if torch.cuda.is_available():
    device = torch.device("cuda:0")

else:
    device = torch.device("cpu")

model = Classifier().to(device)

importing_path = r'assets/'
model.load_state_dict(torch.load(importing_path + 'lightCNN_timeClassifier_Pytorch_preprocessing_state_dict.pth', map_location = device))
model.eval()

with open(importing_path + 'decoder.json', 'rt') as r:
    decoder = json.load(r)


class Item(BaseModel):
    record: list[float]

app = FastAPI()

@app.post("/batch_predict/")
def batch_predict(items: list[Item]):
    st = time.time()
    x = torch.tensor([item.record for item in items]).reshape(-1, 2048)
    x_scaled = torch.autograd.Variable(scaler(x).reshape(-1, 1, 2048).float()).to(device)
    preds = torch.softmax(model(x_scaled), 1)
    
    analysis = [{"Prediction": decoder[str(i.index(max(i)))], "Probabilities": dict(zip(decoder.values(), i))} for i in preds.tolist()]

    et = time.time()

    return {"Analysis": analysis, "Execution Time": 1000 * (et - st)}