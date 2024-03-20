import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import label


app = FastAPI(
    title="Agro Expert",
    description="Provide the soil condition and recommend the crops",
    version="0.0.1",
)

with open('Farm_expert_soil_model.pickle', 'rb') as file:
    modelSoil = pickle.load(file)

modelSoil=modelSoil['soil']


with open('Farm_expert_crop_model.pickle', 'rb') as file:
    modelCrop = pickle.load(file)
    
modelCrop=modelCrop['cropM']



class FarmExpertInput_soil(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    label:str

class FarmExpertInput_crop(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/")
def getting():
    return {"app":"started"}


@app.post("/predict_soil")
def predict_soil(data: FarmExpertInput_soil):
    encoded_label=label.lab_encode[data.label]
    input_data = [[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall, encoded_label]]
    try:
        prediction = modelSoil.predict(input_data)
        return {"status":"true",
                "N":round(prediction[0][0],2),
                "P":round(prediction[0][1],2),
                "K":round(prediction[0][2],2),
                "temperature":round(prediction[0][3],2),
                "humidity":round(prediction[0][4],2),
                "ph":round(prediction[0][5],2),
                "rainfall":round(prediction[0][6],2),
                "label":label.lab_decode[round(prediction[0][7])]
                }
    except:
        return {"prediction": "Error during prediction"}
    
@app.post("/predict_crop")
def predict_crop(data: FarmExpertInput_crop):
    input_data = [[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]]
    try:
        prediction = modelCrop.predict(input_data)
        return {
            "status":"true",
            "N":round(prediction[0][0],2),
            "P":round(prediction[0][1],2),
            "K":round(prediction[0][2],2),
            "temperature":round(prediction[0][3],2),
            "humidity":round(prediction[0][4],2),
            "ph":round(prediction[0][5],2),
            "rainfall":round(prediction[0][6],2),
            "label":label.lab_decode[round(prediction[0][7])]
            }
    except:
        return {"prediction": "Error during prediction"}
