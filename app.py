import uvicorn
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from data_models import PredictionDataset


app = FastAPI()
#  current file path  
current_file_path = Path(__file__).parent

# define paths
model_path = current_file_path / "models" / "models" / "xgboost.joblib"
preprocessor_path = current_file_path / "models" / "transformers" / "preprocessor.joblib"
output_tranformer_path = preprocessor_path.parent / "output_transformer.joblib"

# Load the pre-trained model, preprocessor, and output transformer using joblib
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
output_tranformer = joblib.load(output_tranformer_path)

# Create a pipleline that includes preprocess and the regression model
model_pipleline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])
 # Get -> get some response from api
 # Post -> sending something to API

@app.get('/')
def home():
    return "Welcome to trip duration Prediction app"

@app.post('/predictions')
def do_predictions(test_data: PredictionDataset):
    X_test = pd.DataFrame(
        data = {
            'vendor_id':test_data.vendor_id,
            'passenger_count':test_data.passenger_count,
            'pickup_longitude':test_data.pickup_longitude,
            'pickup_latitude':test_data.pickup_latitude,
            'dropoff_longitude':test_data.dropoff_longitude,
            'dropoff_latitude':test_data.dropoff_latitude,
            'pickup_hour':test_data.pickup_hour,
            'pickup_date':test_data.pickup_date,
            'pickup_month':test_data.pickup_month,
            'pickup_day':test_data.pickup_day,
            'is_weekend':test_data.is_weekend,
            'haversine_distance':test_data.haversine_distance,
            'euclidean_distance':test_data.euclidean_distance,
            'manhattan_distance':test_data.manhattan_distance
        }, index = [0]
    )
    # Make predictions using the pipeline and reshape the result
    predictions = model_pipleline.predict(X_test).reshape(-1, 1)

    # Apply the inverse transformation to the predictions to get the output in original scale
    output_inverse_transformed = output_tranformer.inverse_transform(predictions)[0].item()

    return f"Trip duration for the trip is {output_inverse_transformed:.2f} minutes"


# Main entry point for running the app using uvicorn
if __name__ == "__main__":
    uvicorn.run(app="app:app",
                port=8000)