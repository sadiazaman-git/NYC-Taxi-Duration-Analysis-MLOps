import joblib
import sys
import pandas as pd
from yaml import safe_load
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression



TARGET = 'trip_duration'


def load_dataframe(path):
    df = pd.read_csv(path)
    return df



def make_X_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()

    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]

    return X, y


def train_model(model,X_train,y_train):
    # fit the model on data
    model.fit(X_train,y_train)

    return model


def save_model(model,save_path):
    joblib.dump(value=model,
                filename=save_path)


def main():
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # read input file path
    training_data_path = root_path / sys.argv[1]
    # load the data 
    train_data = load_dataframe(training_data_path)
    # split the data into X and y
    X_train, y_train = make_X_y(dataframe=train_data,target_column=TARGET)
    # read the parameters from params.yaml
    with open('params.yaml') as f:
        params = safe_load(f)

    # Create a dictionary to store models
    models = {
        'random_forest': RandomForestRegressor(**params['train_model']['random_forest_regressor']),
        'xgboost': XGBRegressor(**params['train_model']['xgboost']),
        'gradient_boosting': GradientBoostingRegressor(**params['train_model']['gradient_boosting_regressor']),
        'linear_regression': LinearRegression()
    }
    # Train and save each model
    model_output_path = root_path / 'models'/ 'models'
    model_output_path.mkdir(exist_ok=True)

    for model_name, model in models.items():
        trained_model = train_model(model=model, X_train=X_train, y_train=y_train)
        save_model(model=trained_model, save_path=model_output_path / f'{model_name}.joblib')


if __name__ == "__main__":
    main()