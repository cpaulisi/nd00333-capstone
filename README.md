<img align="right" width="200" height="150" src="https://media.istockphoto.com/photos/wine-pouring-into-glass-picture-id615269202?k=20&m=615269202&s=612x612&w=0&h=S2tAKM7s1kTrJtVKrxmzHNldA_-LRwSgUXirGM_ik20=">

# Prediction of Wine Quality

This project seeks to create a model for classifying wine as either "good" or "bad". A canonical ranking of 5 or below signifies a "bad" wine, while a rating of greater than 5 is considered a target for "good" wine. By using the AutoML and Hyperparameter tuning features within the Azure SDK, a model is generated and deployed for consumption.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
The dataset comprises of the chemical compositions of wines and their subsequent ratings. The target ratings are a classification of either "good" or "bad" (with a canonical score greater than 5 classified as "good"). The dataset was sourced from Kaggle at https://www.kaggle.com/nareshbhat/wine-quality-binary-classification.

### Task
The task is to develop a model that can serve as a binary classifier. It must be able to take the chemical features of a wine as input, and then predict whether or not the wine will recieve a "good" or "bad" score. The chemical features that are utilized as input are fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide,	density, pH, sulphates and 	alcohol.

### Access
The data is accessed via the Azure workspace, using the Workspace and Dataset modules from the SDK. The Workspace is instantiated from a config.json file. Preprocessing is done via temporarily transforming the dataset into a pandas DataFrame.
```python
def preprocess(data):
    y_df = data.pop("quality").apply(lambda x: 1 if x == "good" else 0)
    return data, y_df

ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, name='Wine')
dataset = dataset.to_pandas_dataframe()
x, y = preprocess(dataset)
x['y'] = y
```

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
