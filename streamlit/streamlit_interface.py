import os
import json
import os
import pickle
import time
from typing import Union, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers.trainer_utils import PredictionOutput


class MakeTorchData(torch.utils.data.Dataset):
    """
    Entry data for transformer model needs to be an object of this class
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        item["labels"] = float(item["labels"])
        return item

    def __len__(self):
        return len(self.labels)


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


pd.options.mode.chained_assignment = None

with st.sidebar:
    selection = option_menu(None, ['Data exploration',
                                   'Data analysis from local model',
                                   'Analysis of pre-generated data'],
                            icons=['layers-half', 'hypnotize', 'journal'], default_index=0)

if selection == 'Data exploration':
    st.markdown('''<font size="16"><div style="text-align: center;">Results of data exploration</div></font>''',
                unsafe_allow_html=True)
    data = pd.read_csv('data/recruitment_data_standardized.csv',
                       encoding='utf-8',
                       sep=',',
                       on_bad_lines='skip',
                       quotechar='"',
                       doublequote=True,
                       names=['URL', 'Voivodeship', 'Scrap_time', 'Name', 'Price', 'Brand', 'Condition', 'Offer_from',
                              'Type', 'Description', 'Added_at', 'Views', 'User_since'],
                       skiprows=1)
    st.markdown(
        '''<font size="4"><div style="text-align: center;">Bar plot for amount of NaNs in dataset</div></font>''',
        unsafe_allow_html=True)
    fig = px.bar(pd.DataFrame(data.isna().sum())).update_layout(xaxis_title='Feature name',
                                                                yaxis_title='Number of NaNs',
                                                                showlegend=False)
    st.plotly_chart(fig)
    st.markdown("""---""")
    st.markdown('''<font size="4"><div style="text-align: center;">Percentage of offers in different conditions''',
                unsafe_allow_html=True)
    fig = px.pie(data, names='Condition')
    st.plotly_chart(fig)
    st.markdown("""---""")
    st.markdown('''<font size="4"><div style="text-align: center;">Percentage of offers from different voivodeships''',
                unsafe_allow_html=True)
    fig = px.pie(data, names='Voivodeship')
    st.plotly_chart(fig)
    st.markdown("""---""")
    data['Name'] = data.loc[:, 'Name'].str.lower()
    data['Description'] = data.loc[:, 'Description'].str.lower()
    data['Concatenated_description'] = data['Name'] + ' ' + data['Description']
    data_dropped_price_nans = data.dropna(subset=['Price']).reset_index(drop=True)
    data_dropped_price_nans['Date'] = pd.to_datetime(data_dropped_price_nans['Added_at']).dt.date
    data_reduced_dims = data_dropped_price_nans.query("Condition == 'Używane' & Type == 'Sprawny' & Brand == 'iPhone'")
    data_reduced_dims = data_reduced_dims[data_reduced_dims.Price > 1000]
    data_reduced_dims = data_reduced_dims.drop(
        columns=['Voivodeship', 'Scrap_time', 'Views', 'User_since', 'Added_at', 'URL', 'Brand', 'Condition',
                 'Offer_from',
                 'Type'])
    data_concatenated = data_reduced_dims.drop(columns=['Name', 'Description']).reset_index(drop=True)
    data_concatenated['Phone model'] = None
    for sample in range(len(data_concatenated)):
        if 'iphone 11 pro max' in data_concatenated['Concatenated_description'][sample]:
            data_concatenated['Phone model'][sample] = 'iphone 11 pro max'
        elif 'iphone 11 pro' in data_concatenated['Concatenated_description'][sample]:
            data_concatenated['Phone model'][sample] = 'iphone 11 pro'
        elif 'iphone 11' in data_concatenated['Concatenated_description'][sample]:
            data_concatenated['Phone model'][sample] = 'iphone 11'
    data_grouped = data_concatenated.drop(columns='Concatenated_description').reset_index(drop=True)
    st.markdown(
        '''<font size="4"><div style="text-align: center;">Average prices of different iPhone 11 models with standard deviation''',
        unsafe_allow_html=True)
    fig = px.box(data_grouped, x='Phone model', y='Price')

    st.plotly_chart(fig)
    st.markdown("""---""")

    groups = dict(list(data_grouped.groupby(['Phone model'])['Price', 'Date']))
    st.markdown(
        '''<font size="4"><div style="text-align: center;">Average prices of iPhone 11 base model with standard deviation over time''',
        unsafe_allow_html=True)
    fig = px.box(groups['iphone 11'], x='Date', y='Price')

    st.plotly_chart(fig)
    st.markdown("""---""")
    st.markdown(
        '''<font size="4"><div style="text-align: center;">Average prices of iPhone 11 Pro with standard deviation over time''',
        unsafe_allow_html=True)
    fig = px.box(groups['iphone 11 pro'], x='Date', y='Price',
                 range_x=['2021-01-01', '2021-02-24'])
    st.plotly_chart(fig)
    st.markdown("""---""")
    st.markdown(
        '''<font size="4"><div style="text-align: center;">Average prices of iPhone 11 Pro Max with standard deviation over time''',
        unsafe_allow_html=True)
    fig = px.box(groups['iphone 11 pro max'], x='Date', y='Price',
                 range_x=['2021-01-01', '2021-02-24'])
    st.plotly_chart(fig)
    st.markdown('''<div style="text-align: justify;">  Simple analysis with data exploration, filtering and grouping 
    methods shows that most probably there is no visible relation between price of iPhones and progressing time 
    series for all 3 submodels. This could occur, because there were too few samples in the dataset (especially for 
    Pro and Pro Max models) or perhaps such a relation does not exist in real life. This is a very important insight 
    to have in mind going into NLP model building.</div>''', unsafe_allow_html=True)
    st.markdown("""---""")

    st.markdown('''<font size="4"><div style="text-align: center;">Distplot of prices for different iPhone models''',
                unsafe_allow_html=True)
    fig = ff.create_distplot([list(groups.values())[i]['Price'] for i in range(len(groups.values()))],
                             [list(groups.keys())[i] for i in range(len(groups.keys()))],
                             bin_size=75).update_layout(xaxis_range=(1500, 4500))

    st.plotly_chart(fig)
    st.markdown('''<div style="text-align: justify;">  The disproportion in price for different iPhone 11 models 
    is clear from the distplot. Further dividing the dataset into different memory variant of the models is possible.
    </div>''', unsafe_allow_html=True)


elif selection == 'Data analysis from local model':
    if os.path.isfile('models/auto_model/test_model/pytorch_model.bin'):
        wanna_predict = st.radio('Do you want to perform your own prediction?', ['Yes', 'No'])
        if wanna_predict == 'Yes':
            text_to_predict = st.text_input('Please input text for which the price will be predicted for', 'iPhone 11')
            wanna_ranges = st.radio('Do you want to include time ranges in your prompt?', ['Yes', 'No'])

            if wanna_ranges:
                range_of_days = st.slider(
                    'Select a range of days to prompt',
                    0, 100, (0, 100))

        wanna_metrics = st.radio('Do you want to plot metrics of your own model?', ['Yes', 'No'])

        col1_1, _, col2_1, _, col3_1 = st.columns(5)
        PATH_FOR_MODEL = 'models/auto_model/test_model'
        if col2_1.button('Engage'):
            if wanna_metrics == 'Yes':
                metric_predictions = get_metrics_from_training(
                    trainer_state_path=PATH_FOR_MODEL + '/checkpoints/checkpoint-135/trainer_state.json',
                    scaler_path=PATH_FOR_MODEL + '/scaler.pkl')

                st.markdown(
                    '''<font size="4"><div style="text-align: center;">Metrics for model trained on days + name string for 5 epochs''',
                    unsafe_allow_html=True)
                fig = px.line(metric_predictions[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis=dict(tickmode='linear',
                                                                                                   tick0=0,
                                                                                                   dtick=1
                                                                                                   ),
                                                                                        xaxis_title='Epoch',
                                                                                        yaxis_title='Metric value')

                st.plotly_chart(fig)
                st.markdown("""---""")
                st.markdown(
                    '''<font size="4"><div style="text-align: center;">R2 index for model trained on days + name string for 5 epochs''',
                    unsafe_allow_html=True)
                fig = px.line(metric_predictions[['R2']]).update_layout(xaxis=dict(tickmode='linear',
                                                                                   tick0=0,
                                                                                   dtick=1
                                                                                   ), xaxis_title='Epoch',
                                                                        yaxis_title='Metric value')

                st.plotly_chart(fig)

            if wanna_predict:
                with st.spinner('Performing predictions...'):
                    trainer = load_trainer_for_prediction(PATH_FOR_MODEL)
                    tokenizer = AutoTokenizer.from_pretrained('dkleczek/bert-base-polish-uncased-v1')
                    scaler = pickle.load(open(PATH_FOR_MODEL + '/scaler.pkl', 'rb'))

                    prediction = predict_text(trainer, scaler, tokenizer, text_to_predict, range_of_days)
                    st.markdown("""---""")
                    st.markdown(
                        '''<font size="4"><div style="text-align: center;">Changes in price for the prompt in relation to days passed''',
                        unsafe_allow_html=True)
                    fig = px.scatter(prediction, trendline='lowess',
                                     ).update_layout(
                        xaxis_title='Days passed', yaxis_title='Price', showlegend=False)

                    fig.add_vline(55, annotation_text='Last day in dataset')

                    st.plotly_chart(fig)
    else:
        st.error('Model was not found, please run 04_auto_model.ipynb in src')



elif selection == 'Analysis of pre-generated data':
    which_metrics = st.multiselect(
        'Metrics for which model do you want to see?',
        ['Days + name in 5 epochs',
         'Days + name in 20 epochs',
         'Days + name + part of description in 5 epochs',
         'Days + name + part of description in 20 epochs'])

    which_text = st.multiselect(
        'Predictions for which prompts do you want to see?',
        ['iphone 11 64, 128, 256gb',
         'iphone 11 pro 64, 256, 512gb',
         'iphone 11 pro max 64, 256, 512gb'])

    col1_1, col2_1, col3_1 = st.columns(3)
    if col2_1.button('Engage visualizations'):
        with st.spinner('Loading data in progress'):
            days_name_5_metrics = pd.read_csv('streamlit/metrics/days_name_5_metrics.csv', index_col=0)
            days_name_20_metrics = pd.read_csv('streamlit/metrics/days_name_20_metrics.csv', index_col=0)
            days_name_desc_5_ml100_metrics = pd.read_csv('streamlit/metrics/days_name_desc_5_ml100_metrics.csv', index_col=0)
            days_name_desc_20_ml100_metrics = pd.read_csv('streamlit/metrics/days_name_desc_20_ml100_metrics.csv', index_col=0)
            predictions = pd.read_csv('streamlit/predictions/days_name_5_predictions.csv', index_col=0)
            time.sleep(1)
        st.success('Loading data done!')

        if 'Days + name in 5 epochs' in which_metrics:
            st.markdown("""---""")
            st.markdown(
                '''<font size="4"><div style="text-align: center;">Metrics for model trained on days + name string for 5 epochs''',
                unsafe_allow_html=True)
            fig = px.line(days_name_5_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                     yaxis_title='Metric value')
            st.plotly_chart(fig)

        if 'Days + name in 20 epochs' in which_metrics:
            st.markdown("""---""")
            st.markdown(
                '''<font size="4"><div style="text-align: center;">Metrics for model trained on days + name string for 20 epochs''',
                unsafe_allow_html=True)
            fig = px.line(days_name_20_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                      yaxis_title='Metric value')
            st.plotly_chart(fig)

        if 'Days + name + part of description in 5 epochs' in which_metrics:
            st.markdown("""---""")
            st.markdown(
                '''<font size="4"><div style="text-align: center;">Metrics for model trained on days + name + description string for 5 epochs''',
                unsafe_allow_html=True)
            fig = px.line(days_name_desc_5_ml100_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                                yaxis_title='Metric value')
            st.plotly_chart(fig)

        if 'Days + name + part of description in 20 epochs' in which_metrics:
            st.markdown("""---""")
            st.markdown(
                '''<font size="4"><div style="text-align: center;">Metrics for model trained on days + name + description string for 20 epochs''',
                unsafe_allow_html=True)
            fig = px.line(days_name_desc_20_ml100_metrics[['MSE', 'MAE', 'RMSE']]).update_layout(xaxis_title='Epoch',
                                                                                                 yaxis_title='Metric value')
            st.plotly_chart(fig)

        st.markdown('''<div style="text-align: justify;">  To summarize metrics values changing over time - it is 
        seen that the model is trying to learn, but not a lot of advance is made in this direction. In 2 models 
        during 20 epochs the model did not decrease MSE, MAE or RMSE, though it is possible that my method of inverse 
        scaling it after getting it out of the model may be faulty (the value of RMSE should be root of MSE, 
        but it is not). Quick mean square error calculation at the end of 04_auto_model.ipynb, done on test set vs 
        model prediction outputted MSE ~400, which was more believable than the values above. These error 
        calculations done by the model itself needs further investigation.</div>''', unsafe_allow_html=True)

        if 'iphone 11 64, 128, 256gb' in which_text:
            st.markdown("""---""")
            st.markdown(
                '''<font size="4"><div style="text-align: center;">Changes in price for vanilla variants in relation to days passed since 01.01.2021''',
                unsafe_allow_html=True)
            fig = px.scatter(predictions[['iphone 11 64gb',
                                          'iphone 11 128gb',
                                          'iphone 11 256gb']], trendline='lowess').update_layout(
                xaxis_title='Days passed', yaxis_title='Price')
            fig.add_vline(55, line_color='white', annotation_text='Last day in dataset')
            st.plotly_chart(fig)

        if 'iphone 11 pro 64, 256, 512gb' in which_text:
            st.markdown("""---""")
            st.markdown(
                '''<font size="4"><div style="text-align: center;">Changes in price for Pro variants in relation to days passed since 01.01.2021''',
                unsafe_allow_html=True)
            fig = px.scatter(predictions[['iphone 11 pro 64gb',
                                          'iphone 11 pro 256gb',
                                          'iphone 11 pro 512gb']], trendline='lowess').update_layout(
                xaxis_title='Days passed', yaxis_title='Price')
            fig.add_vline(55, line_color='white', annotation_text='Last day in dataset')
            st.plotly_chart(fig)

        if 'iphone 11 pro max 64, 256, 512gb' in which_text:
            st.markdown("""---""")
            st.markdown(
                '''<font size="4"><div style="text-align: center;">Changes in price for Pro Max variants in relation to days passed since 01.01.2021''',
                unsafe_allow_html=True)
            fig = px.scatter(predictions[['iphone 11 pro max 64gb',
                                          'iphone 11 pro max 256gb',
                                          'iphone 11 pro max 512gb']], trendline='lowess').update_layout(
                xaxis_title='Days passed', yaxis_title='Price')
            fig.add_vline(55, line_color='white', annotation_text='Last day in dataset')
            st.plotly_chart(fig)

        st.markdown('''<div style="text-align: justify;"> It is wonderful to be able to say that the model seems to have learned to differentiate iPhone \
                    11 models based on price (even including different memory variants). It is clearly visible that \
                    predictions of prices for "Max" prompts are of greater value than "Pro" prompts and those \
                    are above in price in regards to usual "iphone 11" prompts. The prices are also higher for \
                    models with more memory for each iPhone 11 type.</div>''', unsafe_allow_html=True)
        f''
        st.markdown('''<div style="text-align: justify;"> 
                    On the other hand the model does not seem to grasp time series relevance - after crossing the \
                    barrier of 55 days (up to which samples are present in dataset) the model always flattens out price values, \
                    no matter what was the trend before the barrier. It is also worth mentioning that there is a \
                    difference in price prediction standard deviation in phrases like "64" vs "64gb" - the \
                    latter, which is probably more common in the dataset, outputs predictions with considerably less \
                    standard deviation (example with the "64" phrase is in 04_auto_model.ipynb notebook).</div>''',
                    unsafe_allow_html=True)
