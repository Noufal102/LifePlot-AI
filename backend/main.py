
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database import Base, engine, get_db
from models import Activity, Prediction
from schemas import ActivityCreate, ActivityOut, PredictionOut
from predictor import predict_next
from data_utils import ACTIVITIES
from decision_engine import generate_daily_insights


# =====================================
# DATABASE INITIALIZATION
# =====================================

Base.metadata.create_all(bind=engine)


# =====================================
# FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="LifePlot AI API",
    version="1.4.0",
    description="Transformer-Based Real-Time Daily Activity Prediction & Decision Assistant"
)


# =====================================
# CORS CONFIGURATION
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================
# ROOT API
# =====================================

@app.get("/")
def root():
    return {
        "message": "LifePlot AI API is running",
        "status": "success",
        "version": "1.4.0"
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "database": "SQLite",
        "model": "Transformer Encoder"
    }


# =====================================
# AVAILABLE ACTIVITY TYPES
# =====================================

@app.get("/api/activities/types")
def activity_types():
    return {
        "activities": ACTIVITIES
    }


# =====================================
# ADD ACTIVITY
# =====================================

@app.post("/api/activities", response_model=ActivityOut)
def add_activity(
    payload: ActivityCreate,
    db: Session = Depends(get_db)
):
    if payload.activity not in ACTIVITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Choose one of: {', '.join(ACTIVITIES)}"
        )

    item = Activity(**payload.model_dump())

    db.add(item)
    db.commit()
    db.refresh(item)

    return item


# =====================================
# GET ACTIVITY HISTORY
# =====================================

@app.get("/api/activities", response_model=list[ActivityOut])
def get_activities(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    limit = max(1, min(limit, 500))

    return (
        db.query(Activity)
        .order_by(desc(Activity.timestamp))
        .limit(limit)
        .all()
    )


# =====================================
# DELETE ACTIVITY
# =====================================

@app.delete("/api/activities/{activity_id}")
def delete_activity(
    activity_id: int,
    db: Session = Depends(get_db)
):
    item = (
        db.query(Activity)
        .filter(Activity.id == activity_id)
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Activity not found"
        )

    db.delete(item)
    db.commit()

    return {
        "message": "Activity deleted",
        "deleted_id": activity_id
    }


# =====================================
# TRANSFORMER PREDICTION
# WITH TOP 3 RESULTS
# =====================================

@app.get("/api/predict", response_model=PredictionOut)
def get_prediction(
    db: Session = Depends(get_db)
):

    history = (
        db.query(Activity)
        .order_by(Activity.timestamp.asc())
        .all()
    )

    result = predict_next(history)

    # Save only genuine predictions.
    # Do not save Not Ready responses.

    predicted_activity = result.get("predicted_activity")

    if predicted_activity not in (None, "", "Not Ready"):

        record = Prediction(
            predicted_activity=predicted_activity,
            confidence=result["confidence"],
            suggestion=result["suggestion"]
        )

        db.add(record)
        db.commit()

    return result


# =====================================
# ACTIVITY ANALYTICS
# =====================================

@app.get("/api/analytics")
def analytics(
    db: Session = Depends(get_db)
):

    items = db.query(Activity).all()

    totals = {}

    for item in items:
        totals[item.activity] = (
            totals.get(item.activity, 0) + item.duration
        )

    return {
        "total_activities": len(items),
        "total_minutes": round(
            sum(x.duration for x in items), 1
        ),
        "duration_by_activity": totals
    }


# =====================================
# PREDICTION HISTORY
# =====================================

@app.get("/api/predictions")
def prediction_history(
    limit: int = 20,
    db: Session = Depends(get_db)
):

    limit = max(1, min(limit, 100))

    rows = (
        db.query(Prediction)
        .order_by(desc(Prediction.created_at))
        .limit(limit)
        .all()
    )

    return [
        {
            "predicted_activity": x.predicted_activity,
            "confidence": x.confidence,
            "suggestion": x.suggestion,
            "created_at": x.created_at.isoformat()
        }
        for x in rows
    ]


# =====================================
# MODEL PERFORMANCE
# =====================================

@app.get("/api/model-performance")
def model_performance():

    return {
        "transformer": {
            "accuracy": 0.3011,
            "macro_f1": 0.2555
        },
        "random_forest": {
            "macro_f1": 0.2644
        },
        "dataset": "Synthetic evaluation dataset",
        "note": "Average evaluation results across multiple seeds"
    }


# =====================================
# DAILY INSIGHTS
# =====================================

@app.get("/api/daily-insights")
def daily_insights(
    db: Session = Depends(get_db)
):

    history = (
        db.query(Activity)
        .order_by(Activity.timestamp.asc())
        .all()
    )

    return generate_daily_insights(history)
