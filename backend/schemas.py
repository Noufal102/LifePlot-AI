
from pydantic import BaseModel, Field
from datetime import datetime


class ActivityCreate(BaseModel):
    activity: str = Field(min_length=2, max_length=40)
    duration: float = Field(gt=0, le=1440)
    energy_level: int = Field(ge=1, le=5)
    priority: int = Field(ge=1, le=5)


class ActivityOut(BaseModel):
    id: int
    activity: str
    duration: float
    energy_level: int
    priority: int
    timestamp: datetime

    class Config:
        from_attributes = True


class TopPrediction(BaseModel):
    activity: str
    confidence: float


class PredictionOut(BaseModel):
    predicted_activity: str
    confidence: float
    suggestion: str
    top_predictions: list[TopPrediction] = []
