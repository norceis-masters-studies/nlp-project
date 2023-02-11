# <center> Predicting phone price based on processing text from trade offers

## <center>Project summary
<justify>The goal of the project was to extrapolate the price of iPhones into the future from data scrapped from OLX of 
time period spanning almost 2 months. The project was divided into six parts: data standardization, data exploration, 
data preprocessing, NLP model, data visualization and user interface with visualized data. These parts are described in 
detail below. All of the notebooks in src dir contain detailed comments about the purpose of the code. Some parts of 
the code were functionalized and moved to **util.py**.

To summarize the results - early exploration od the data has shown that price-time correlation may not be 
present in the data, which is also denoted by the NLP model used in this project. Price-time predictions 
of the model do not seem to correlate strongly with each other. Reasonably I see two potential causes:

1. Model does not work correctly (need better model)
2. The data does not contain any price-time correlation (need bigger dataset)

I think both reasons are valid. Exploring another architecture of the model, possibly better suited
for time series prediction, would be a good direction for the future.

What is interesting though, the model had no issue with detecting price differences in various iPhone 11 variants 
(different memory storage, premium models), which emerged from the predictions in a clear-cut way, even with so few 
epochs and very few premium model samples to train on.  
</justify>
## <center> Project structure
1. src - working files (jupyter notebooks and .py function files)
2. data - csv files with scrapped and processed data
3. models - machine learning models files
4. streamlit - API for user interface with presented data

## <center> Chronological order of working files in src
1. data_standardization.ipynb
2. data_preprocessing.ipynb
3. auto_model.ipynb
4. data_visualization.ipynb
5. data_exploration.ipynb

Code from those notebooks is distilled into functions in **util.py** and **classes.py**

## <center> Part 1 - Data standardization

Data supplied along the task (**recruitment_task.csv**) had samples mostly in 2 general forms:
1. Samples with features separated by a comma with large text features such as "Name" and "Description" having quotes inside of them 
    as a "quotechar", which forbids data processing tools to treat commas inside those texts as separators.
2. Samples with features separated by a comma, but also enclosed in a quote character from start to finish
3. Outlier samples that do not belong to first 2 groups (around 100 samples, discarded)

Having those 2 differently formatted sample groups I needed to standardize them into one format - which is the first one.
is done in **data_standardization.ipynb**, also moving the finished function to **util.py**.

## <center> Part 2 - Data exploration and analysis 
During this step I toyed with plotly possibilites and managed to group data into different iPhone 11 models in order to check
if progressing time series affects the prices. The results of this step are summarized in **02_data_exploration.ipynb** 
and also are available in Streamlit interface for preview. This step uses code that I came up with in the next part - 
my reasoning is detailed there more thoroughly. 


## <center> Part 3 - Data preprocessing
After tackling the problem of standardizing and exploration of data I have started to transform the data according to the 
problem specification **data_preprocessing.ipynb**:
1. At first I check visually if the structure of the data is correct after standardization
2. I study how many empty fields are in the data and also I look at different values and their
   respective count in given column
3. The first preprocessing step is dropping all samples where _'Price'_ is NaN, because those
   are insignificant in teaching the model
4. I filter the data to have samples with Condition==_'Używany'_, Type==_'Sprawny'_ 
   and Brand==_'iPhone'_ according to the specification of the problem. The dataset contains approximately 2/3 of 'Używane'
   samples, so this is the biggest sample cut off. Type and Brand filters only affect no more than few hundred samples
5. At this point I decide to drop all the columns that do not carry important information according 
   to the specification of the problem, e.g. 'Voivodeship' or 'URL'
6. It is also important to cut off samples that do not offer actual product, such as phone case or a repair service - I apply this by dropping all samples below value 1000 in 'Price'.
   This is probably not the best way to do this, but I assumed that any working iPhone 11 model wouldn't be offered
   for such a low price
7. Converting all strings in the data to lowercase
8. Creating new column that counts how many days passed since earliest date in the data
9. Concatenating few columns into one, in order to pass only 1 column to the model later on

Both notebooks resulted in two functions (described in detail in **util.py**) used in latter notebooks. 
Data preprocessing could be more generalized and parametrized.

## <center> Part 4 - NLP model
Having processed data in form of concatenated text used to feed the model in one column and target price in the other
I proceed to build a basic model with pretrained polish language model _'dkleczek/bert-base-polish-uncased-v1'_ 
from Hugging Face transformers library, which is based on BERT model from Google. 

Preparation of the data for the model consists of several parts:
1. Scaling _'Price'_ values to range (-1, 1)
2. Splitting the dataset into training (64%), validation (16%) and test (20%) groups
3. Tokenization of each of the sets
4. Converting tokens into torch-type dataset along with the labels

Torch-type dataset is the entry to training and validation phase of the process. During training (and at the end of it) at every epoch model metrics and parameters are saved.
This allows us to later load the model and performs predictions with it.

## <center> Part 5 - Data visualisation 
**data_visualization.ipynb** contains both changes in metric values and possible predictions for any model trained
before on the dataset. Metrics used to measure performance of the model are (compute_metrics in **util.py**): 
1. mean square error
2. root mean square error
3. mean absolute error
4. r2 score

This part also contains my observations and thoughts about the model and results in the form of markdown comments.

## <center> Part 6 - Interface with results 
Data acquired during work on the project is presented in Streamlit interface. To launch it click here or run it in terminal (make sure you are in root dir):

```python -m streamlit run streamlit/streamlit_interface.py```

In order to be able to interact with the model itself in Streamlit, you need to generate model files first (run 04_auto_model.ipynb)









