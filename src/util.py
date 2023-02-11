import csv
import json
import pickle
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers.trainer_utils import PredictionOutput

from classes import MakeTorchData


def create_standardized_data(path_to_file: Optional[str] = '../data/recruitment_data.csv',
                             column_number: Optional[int] = None) -> None:
    """
    This function goes through each line in the dataset via "next" generator and depending
    on the sample rewrites it, so the data is standardized (explained in detail in readme.md - part 1)
    :param path_to_file: specify path to csv file with data to standardize
    :param column_number: if this is not specified the function will get this automatically from header
    """

    rows = []
    with open(path_to_file, 'r', encoding='utf-8') as opened_file:
        reader = csv.reader(opened_file)
        while True:
            try:
                line = next(reader)
                if len(line) == 1:
                    line = line[0]  # because some samples were enclosed in additional quote we need to
                    rows.append(line)  # get it out of the one-element list it was put into by csv module
                else:
                    rows.append(line)  # rest of the samples were not modified by csv
            except StopIteration:
                break

    if not column_number:
        column_number = len(rows[0])  # number of features present in header of csv file

    with open(path_to_file[:-4] + 'standardized.csv', 'w', encoding='utf-8') as writing_file:

        for sample in rows:
            if len(sample) > column_number:
                try:
                    writing_file.write(sample + '\n')  # outlier samples (n_features > 13) caused problems
                except TypeError:
                    writing_file.write(','.join(sample) + '\n')  # here, this solution works correctly
            else:
                writing_file.write(','.join(sample) + '\n')


def preprocess_data(data: pd.DataFrame,
                    columns_to_drop: Union[None, list] = None,
                    price_clip: int = 1000) -> pd.DataFrame:
    """
    This function summarizes the exploration and preprocessing work done in 03_data_preprocessing.ipynb.
    Due to the nature of the problem in this project it was suited for this particular dataset, but it could be
    improved in more general form.
    :param data: DataFrame with the raw data
    :param columns_to_drop: optional, which columns to drop in order to have more compact DataFrame
    :param price_clip: optional, used to cut off samples under this treshold in 'Price'
    :return: processed DataFrame with 'Price', concatenated 'Days_passed', 'Name', 'Description' columns
    """

    # dropping nans in price
    data_dropped_price_nans = data.dropna(subset=['Price']).reset_index(drop=True)

    # adding column with date only
    data_dropped_price_nans['Date'] = pd.to_datetime(data_dropped_price_nans['Added_at']).dt.date

    # clipping samples for our needs
    data_reduced_dims = data_dropped_price_nans.query("Condition == 'Używane' & Type == 'Sprawny' & Brand == 'iPhone'")

    # dropping insignificant columns
    if columns_to_drop is None:
        columns_to_drop = ['Voivodeship', 'Scrap_time', 'Views',
                           'User_since', 'Added_at', 'URL', 'Brand', 'Condition',
                           'Offer_from', 'Type']

    data_reduced_dims = data_reduced_dims.drop(columns=columns_to_drop)

    # filtering phone cases, offered services, other phones based on price
    data_filtered = data_reduced_dims[data_reduced_dims.Price > price_clip]

    # applying lowercase to all letters in Name and Description
    data_filtered.loc[:, 'Name'] = data_filtered.loc[:, 'Name'].str.lower()
    data_filtered.loc[:, 'Description'] = data_filtered.loc[:, 'Description'].str.lower()

    # adding column with days passed since first date
    data_filtered['First_date'] = data_filtered['Date'].min()
    data_filtered['Date'] = pd.to_datetime(data_filtered.Date, format='%Y-%m-%d')
    data_filtered['First_date'] = pd.to_datetime(data_filtered.First_date, format='%Y-%m-%d')
    data_filtered['Days_passed'] = data_filtered['Date'] - data_filtered['First_date']
    data_filtered['Days_passed_name'] = data_filtered['Days_passed'].astype(str) + ' ' + data_filtered['Name']
    data_filtered['Days_passed_name_desc'] = data_filtered['Days_passed'].astype(str) + ' ' + \
                                             data_filtered['Name'] + ' ' + data_filtered['Description']

    # dropping insignificant columns
    data_concatenated = data_filtered.drop(columns=['Name', 'Description',
                                                    'First_date', 'Date', 'Days_passed']).reset_index(drop=True)

    return data_concatenated


def compute_metrics(eval_pred: PredictionOutput) -> Dict:
    """
    This functions returns a dictionary of metrics, that is used in torch models for checkpoints, but also can be used
    manually to test data predicted by the model.
    :param eval_pred: Data prediction outputted by torch model
    :return: Dictionary of metrics
    """
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def predict_text(model: Trainer,
                 scaler: StandardScaler,
                 tokenizer: AutoTokenizer,
                 text: Union[str, list] = 'iphone 11',
                 days: Union[tuple, None] = (1, 90)) -> np.ndarray:
    """
    This function returns 'Price' predictions of a given text in a time period
    :param model: Trainer object, trained on the dataset
    :param scaler: Scaler, fitted to the dataset, it's needed for inversion of data scaling
    :param tokenizer: Tokenizer object, used to tokenize the dataset in order to be predicted by model
    :param text: Text to be processed; if it's string number of days passed will be added from range in :param days,
                 If it's list of strings samples will be unchanged
    :param days: Range of days (equal to number of samples in this prediction dataset) that will be added to the text
                 If None, string of days passed will not be added to text
    :return: Numpy array with 'Price' predictions (rising number of days equals to time progression)
    """

    samples_to_predict = []

    if isinstance(text, str):
        text = [text]

    if not days:
        samples_to_predict += [i for i in text]
    else:
        samples_to_predict += [
            f'{days_passed} days {sample}'
            for days_passed in range(*days)
            for sample in text
        ]

    placeholder_prices = [0 for _ in range(len(samples_to_predict))]

    tokens = tokenizer(samples_to_predict, truncation=True, padding=True, max_length=50)
    dataset = MakeTorchData(tokens, np.asarray(placeholder_prices).ravel())
    predictions = model.predict(dataset)
    prices = scaler.inverse_transform(np.asarray(predictions[0]).reshape(-1, 1))

    return prices


def get_metrics_from_training(trainer_state_path: str,
                              scaler_path: str) -> pd.DataFrame:
    """
    Extracts metric values for each training epoch in model
    :param trainer_state_path: trainer state path contains metric values
    :param scaler_path: scaler path, used for inverse scaling of the metrics
    :return: DataFrame with metrics, ready to visualize
    """
    data = json.load(open(trainer_state_path))
    scaler = pickle.load(open(scaler_path, 'rb'))

    mse, mae, rmse, r2 = [], [], [], []
    for epoch in range(len(data['log_history'])):
        mse.append(scaler.inverse_transform(np.asarray(data['log_history'][epoch]['eval_mse']).reshape(-1, 1))[0, 0])
        mae.append(scaler.inverse_transform(np.asarray(data['log_history'][epoch]['eval_mae']).reshape(-1, 1))[0, 0])
        rmse.append(scaler.inverse_transform(np.asarray(data['log_history'][epoch]['eval_rmse']).reshape(-1, 1))[0, 0])
        r2.append(data['log_history'][epoch]['eval_r2'])

    metrics = pd.DataFrame(mse, columns=['MSE'])
    metrics['MAE'] = mae
    metrics['RMSE'] = rmse
    metrics['R2'] = r2

    return metrics


def load_trainer_for_prediction(model_path: str) -> Trainer:
    """
    Returns trainer ready for prediction
    :param model_path: Path to the folder, where the model resides (not the actual model itself!)
    :return: trainer object ready for prediction
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=20,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='../logs',
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model='rmse',
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    return trainer
