from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class Activity(Base):
    __tablename__ = "activities"
    id = Column(Integer, primary_key=True, index=True)
    activity = Column(String, nullable=False)
    duration = Column(Float, nullable=False)
    energy_level = Column(Integer, default=3)
    priority = Column(Integer, default=3)
    timestamp = Column(DateTime, default=datetime.now, index=True)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    predicted_activity = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    suggestion = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
