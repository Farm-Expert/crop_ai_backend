# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# app=FastAPI(title="Farm Exprt", description="Provide the soil condition and recommend the crops",version="0.0.1")

# # schema
# class FarmExpertInput(BaseModel):
#     N=float
#     P=float
#     K=float
#     temperature=float
#     humidity=float
#     ph=float
#     rainfall=float
#     # label=float


# def predict(data:FarmExpertInput):
#     model=joblib.load('Farm_expert_model.pickle')
#     data_in1=[[data.N,data.P,data.K,data.temperature,data.humidity,data.ph,data.rainfall]]
#     # data_in2=[[N,P,K,temperature,humidity,ph,rainfall,label]]
#     prediction=model.predict(data_in1)

#     # prediction1=model2.predict()
#     prob=model.predict_proba([data_in1]).max()
#     return(prediction,prob)


# @app.post("/predict")
# def predict(farm:FarmExpertInput):
#     farm.model_dict()
#     pred,pro=predict(farm)
#     return{
#         "preduction":pred.tolist(),
#         "probability":pro
#     }

# ---------------
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from label import lab
import pickle
# import numpy
# print(numpy.version.version)

# C:\Users\Dell\OneDrive\Documents\fastapi\farm_expert\main.py
# curl https://bootst/rap.pypa.io/get-pip.py -o get-pip.py

# label=lab[input]

app = FastAPI(
    title="Farm Expert",
    description="Provide the soil condition and recommend the crops",
    version="0.0.1",
)


# read of picklefile
f=open('Farm_expert_soil_model.pickle','rb')
model = load(f)
label:str
missing_go_to_left = 1
# Pydantic model

#crop model
class FarmExpertInput_soil(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    label:str

# Function to predict
def predict_crop(data: FarmExpertInput_soil):
    data_in = [[
        data.N, data.P, data.K, data.temperature,
        data.humidity, data.ph, data.rainfall,data.label
    ]]
    prediction = model.predict(data_in)
    return prediction

@app.post("/predict")
def predict_soil(farm_input: FarmExpertInput_soil):
    pred= predict_soil(farm_input)
    return {
        "prediction": pred.tolist(),  # Convert to a Python list
    }
# ----------------

# soil

f=open('Farm_expert_crop_model.pickle','rb')
model = load(f)
class FarmExpertInput_crop(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

def predict_soil(data: FarmExpertInput_crop):
    
    data_in = [[
        data.N, data.P, data.K, data.temperature,
        data.humidity, data.ph, data.rainfall
    ]]
    prediction = model.predict(data_in)
    # prob = model.predict_proba(data_in).max()
    return prediction
# Endpoint to receive input and return prediction
@app.post("/predict")
def predict_soil(farm_input: FarmExpertInput_crop):
    pred= predict_soil(farm_input)
    return {
        "prediction": pred.tolist(),  # Convert to a Python list
    }
