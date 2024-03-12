from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_pipeline
import pickle
import re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



app = FastAPI()

# Allow all origins, allow all methods, allow all headers
# You might want to adjust these settings based on your requirements
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Features(BaseModel):
    gender: str
    NationalITy: str
    PlaceofBirth: str
    StageID: str
    GradeID: str
    SectionID: str
    Topic: str
    Semester: str
    Relation: str
    raisedhands: int
    VisITedResources: int
    AnnouncementsView: int
    Discussion: int
    ParentAnsweringSurvey: str
    ParentschoolSatisfaction: str
    StudentAbsenceDays: str




@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict" )
def predict(features: Features):
    predicted_class = predict_pipeline(features)

    return {"Class": predicted_class}