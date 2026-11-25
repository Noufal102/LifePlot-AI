# LifePlot AI – Transformer-Based Real-Time Daily Activity Prediction & Decision Assistant

LifePlot AI is an AI-powered daily activity prediction and decision-support application. It analyzes logged activities and uses a Transformer Encoder model to predict the next likely activity and generate meaningful daily insights.

## 1. Project Overview

Managing daily activities and understanding personal routines can be challenging. LifePlot AI aims to help users track their activities, understand their daily patterns, and receive AI-assisted suggestions.

The application combines a browser-based dashboard, a FastAPI backend, machine learning inference, and an SQLite database.

## 2. Key Features

- **Activity Management:** Add, view, and delete daily activity records.
- **Activity Prediction:** Predict the next likely activity using a Transformer Encoder.
- **Prediction History:** View previously generated predictions.
- **Daily Insights:** Summarize activity duration, energy levels, and high-priority activities.
- **AI Recommendations:** Generate suggestions based on recorded activity data.
- **Model Performance:** Display evaluation metrics for Transformer and Random Forest models.
- **Analytics Dashboard:** Visualize activity statistics and duration distributions.
- **Persistent Storage:** Store activities and predictions using SQLite.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Backend | Python, FastAPI |
| Machine Learning | PyTorch |
| Model Architecture | Transformer Encoder |
| Baseline Model | Random Forest |
| Database | SQLite |
| ORM | SQLAlchemy |
| API Validation | Pydantic |

## 4. System Architecture

```text
              USER
                |
                v
      HTML / CSS / JavaScript
             Frontend
                |
           HTTP Requests
                |
                v
          FastAPI Backend
                |
       +--------+---------+
       |        |         |
       v        v         v
   Activity  Predictor  Decision
  Management  Module     Engine
                |
                v
       Transformer Encoder
                |
                v
        Prediction Results
                |
                v
         SQLite Database
                |
                v
       Dashboard & Insights
```

## 5. Project Structure

```text
LifePlot_AI/
|
|-- backend/
|   |-- main.py
|   |-- database.py
|   |-- models.py
|   |-- schemas.py
|   |-- data_utils.py
|   |-- transformer_model.py
|   |-- train.py
|   |-- predictor.py
|   |-- decision_engine.py
|   |-- requirements.txt
|   |-- activity_transformer_balanced_seed_42.pth
|
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- app.js
|
|-- README.md
```

*The structure above describes the main project files. Additional database, model, or configuration files may also be present.*

## 6. Installation and Setup

### Requirements

- Python 3.10 or later
- Visual Studio Code
- Modern web browser

Open the LifePlot AI project folder in VS Code.

### Step 1: Create a Virtual Environment

Open the terminal:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Start the Backend

From the backend directory:

```powershell
uvicorn main:app --reload
```

Backend URL:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### Step 4: Start the Frontend

Open a second terminal from the project root:

```powershell
cd frontend
python -m http.server 5500
```

Open the application:

http://127.0.0.1:5500/index.html

Alternatively, use the VS Code Live Server extension with `frontend/index.html`.

## 7. Transformer Model and Checkpoint

LifePlot AI uses a Transformer Encoder model for sequential activity prediction.

The current prediction module (`predictor.py`) loads the following checkpoint:

```text
backend/activity_transformer_balanced_seed_42.pth
```

This checkpoint must be available in the backend directory for Transformer prediction.

### Optional Demo Training

The project includes `train.py`, which generates a synthetic sequence dataset for demonstrating the model training pipeline.

Run from the backend directory:

```powershell
python train.py
```

The demo trainer saves:

```text
activity_transformer.pth
```

**Important:** The demo training output and the checkpoint used by the current prediction module have different filenames. Running `train.py` does not replace or generate `activity_transformer_balanced_seed_42.pth`. The existing prediction checkpoint should be preserved.

The included synthetic dataset demonstrates model training, saving, loading, and inference workflows. It does not establish real-world predictive accuracy.

## 8. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/api/health` | Check backend and database health |
| GET | `/api/activities/types` | Get available activity types |
| POST | `/api/activities` | Add a new activity |
| GET | `/api/activities` | Retrieve activity records |
| DELETE | `/api/activities/{activity_id}` | Delete an activity |
| GET | `/api/predict` | Generate the next activity prediction |
| GET | `/api/analytics` | Retrieve activity analytics |
| GET | `/api/daily-insights` | Generate daily insights |
| GET | `/api/predictions` | Retrieve prediction history |
| GET | `/api/model-performance` | Retrieve model evaluation metrics |

## 9. Machine Learning and Model Performance

### Transformer Encoder

The Transformer Encoder processes activity sequences and their associated features to predict a likely next activity.

Model configuration:

- Number of activity classes: 8
- Input features: 5
- Embedding dimension: 64
- Attention heads: 4
- Encoder layers: 2
- Sequence length: 12
- Feed-forward dimension: 128

A minimum of 12 activity records is required for personalized Transformer prediction.

### Random Forest Baseline

A Random Forest model is also included as a baseline for model comparison.

### Evaluation Results

| Model | Metric | Score |
|---|---|---:|
| Transformer | Accuracy | 30.11% |
| Transformer | Macro F1 | 25.55% |
| Random Forest | Macro F1 | 26.44% |

**Important:** These results come from a synthetic evaluation dataset and represent average evaluation results across multiple seeds. They should not be interpreted as proven real-world personal activity prediction accuracy. The current Random Forest baseline has a slightly higher Macro F1 score than the Transformer.

## 10. Testing the Application

1. Start the backend server.
2. Start the frontend server.
3. Open the LifePlot AI dashboard.
4. Add at least 12 activities to enable personalized Transformer prediction.
5. Verify that activity counts and duration statistics update.
6. Generate a next-activity prediction.
7. Check the Top 3 predictions and confidence values.
8. Check prediction history.
9. Review daily insights and AI recommendations.
10. Delete a test activity and verify that the dashboard updates.
11. Check the model performance section.

## 11. Important ML Limitations

The demo training script uses synthetic sequential activity patterns. It demonstrates the training, saving, loading, and inference workflow, but does not prove that the model can accurately predict real personal routines.

For more credible real-world evaluation, future development should include:

- Collecting consented real-world activity logs.
- Defining a clear next-activity prediction target and time window.
- Splitting data chronologically into training, validation, and test sets.
- Comparing the Transformer against simple baselines and Random Forest.
- Reporting held-out accuracy, Macro F1, and per-class performance.
- Evaluating prediction latency and reliability.
- Improving the model using additional real-world data.

## 12. Future Scope

- Real-time activity monitoring.
- Personalized recommendations.
- Improved model performance using real-world datasets.
- Model comparison and experiment tracking.
- Advanced routine analytics.
- User authentication and personalized profiles.

## 13. Project Objective

The main objective of LifePlot AI is to combine machine learning, sequential activity prediction, and decision-support features into one application that helps users understand and manage their daily routines.

---

**LifePlot AI – Making Daily Activities Smarter with AI.**