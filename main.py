from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from typing import List, Optional

app = FastAPI(
    title="Student Exam Score Prediction API",
    description="ML API for predicting student exam scores",
    version="1.0"
)


# Student data model
class StudentData(BaseModel):
    hours_studied: float
    previous_scores: float
    attendance: float
    extracurricular: float
    parental_education: float


class TrainingConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    max_depth: Optional[int] = None
    min_samples_split: int = 2


# Global variables to store models and metrics
linear_model = None
tree_model = None
linear_metrics = {}
tree_metrics = {}
current_dataset = None


def generate_student_data(num_students=500):
    """Generate synthetic student data for demonstration"""
    np.random.seed(42)

    data = {
        'hours_studied': np.random.normal(5, 2, num_students).clip(1, 10),
        'previous_scores': np.random.normal(75, 15, num_students).clip(40, 100),
        'attendance': np.random.normal(85, 10, num_students).clip(60, 100),
        'extracurricular': np.random.normal(2, 1, num_students).clip(0, 5),
        'parental_education': np.random.choice([1, 2, 3, 4, 5], num_students, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    }

    # Generate target variable
    base_score = (
            data['hours_studied'] * 2.5 +
            data['previous_scores'] * 0.4 +
            data['attendance'] * 0.3 +
            data['parental_education'] * 1.5 -
            data['extracurricular'] * 0.8 +
            np.random.normal(0, 5, num_students)
    )

    data['final_score'] = base_score.clip(0, 100)

    return pd.DataFrame(data)


@app.on_event("startup")
async def startup_event():
    """Generate initial dataset on startup"""
    global current_dataset
    current_dataset = generate_student_data()


@app.get("/")
async def root():
    return {
        "message": "Student Exam Score Prediction API",
        "endpoints": {
            "train": "POST /train - Train ML models",
            "predict": "POST /predict - Predict exam score",
            "data": "GET /data - Get sample data",
            "metrics": "GET /metrics - Get model performance"
        }
    }


@app.post("/train")
async def train_models(config: TrainingConfig):
    """Train both Linear Regression and Decision Tree models"""
    global linear_model, tree_model, linear_metrics, tree_metrics, current_dataset

    try:
        # Prepare features and target
        X = current_dataset[['hours_studied', 'previous_scores', 'attendance', 'extracurricular', 'parental_education']]
        y = current_dataset['final_score']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state
        )

        # Train Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_linear = linear_model.predict(X_test)

        linear_metrics = {
            "r2_score": round(r2_score(y_test, y_pred_linear), 3),
            "mae": round(mean_absolute_error(y_test, y_pred_linear), 2)
        }

        # Train Decision Tree
        tree_model = DecisionTreeRegressor(
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            random_state=config.random_state
        )
        tree_model.fit(X_train, y_train)
        y_pred_tree = tree_model.predict(X_test)

        tree_metrics = {
            "r2_score": round(r2_score(y_test, y_pred_tree), 3),
            "mae": round(mean_absolute_error(y_test, y_pred_tree), 2)
        }

        return {
            "status": "Training completed successfully!",
            "linear_regression_metrics": linear_metrics,
            "decision_tree_metrics": tree_metrics,
            "training_samples": len(X_train),
            "testing_samples": len(X_test)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict")
async def predict_score(student: StudentData):
    """Predict exam score for a student"""
    if linear_model is None or tree_model is None:
        raise HTTPException(status_code=400, detail="Models not trained yet. Please train models first.")

    try:
        # Prepare input data
        input_data = [[
            student.hours_studied,
            student.previous_scores,
            student.attendance,
            student.extracurricular,
            student.parental_education
        ]]

        # Make predictions
        linear_pred = linear_model.predict(input_data)[0]
        tree_pred = tree_model.predict(input_data)[0]

        return {
            "linear_regression_prediction": round(linear_pred, 2),
            "decision_tree_prediction": round(tree_pred, 2),
            "input_data": student.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/data")
async def get_data(sample_size: int = 10):
    """Get sample data from the dataset"""
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="No data available")

    sample_data = current_dataset.head(sample_size)
    return {
        "sample_data": sample_data.to_dict(orient='records'),
        "total_students": len(current_dataset),
        "columns": list(current_dataset.columns)
    }


@app.get("/metrics")
async def get_metrics():
    """Get current model performance metrics"""
    if not linear_metrics or not tree_metrics:
        raise HTTPException(status_code=400, detail="No metrics available. Please train models first.")

    return {
        "linear_regression": linear_metrics,
        "decision_tree": tree_metrics
    }


@app.post("/generate_new_data")
async def generate_new_data(num_students: int = 500):
    """Generate new synthetic dataset"""
    global current_dataset
    current_dataset = generate_student_data(num_students)

    return {
        "message": f"New dataset generated with {num_students} students",
        "dataset_info": {
            "total_students": len(current_dataset),
            "columns": list(current_dataset.columns)
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)