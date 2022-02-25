<img align="right" width="200" height="150" src="https://media.istockphoto.com/photos/wine-pouring-into-glass-picture-id615269202?k=20&m=615269202&s=612x612&w=0&h=S2tAKM7s1kTrJtVKrxmzHNldA_-LRwSgUXirGM_ik20=">

# Prediction of Wine Quality

This project seeks to create a model for classifying wine as either "good" or "bad". A canonical ranking of 5 or below signifies a "bad" wine, while a rating of greater than 5 is considered a target for "good" wine. By using the AutoML and Hyperparameter tuning features within the Azure SDK, a model is generated and deployed for consumption.

## Project Set Up and Installation
The wine.csv file, located in the wine.zip file, must be uploaded into the Azure account blob storage and name "Wine".
The entry script (echo_score.py) must be uploaded to the cloud along with the notebooks.
After the AutoML model is deployed, authorization is enabled and the respective API key for the webservice but be assigned to the "api_key" variable in the AutoML notebook like below:
```python
api_key = 'XXXXXXXXXXXXXXXXXXXX'
```
The project setup also requires you to download and then reupload a "config.json" file from Azure into the "training" folder created by a cell in the hyperdrive run notebook. This action is recommended after the training.py script is written, but before the hyperdrive run is defined and activated in the workspace. 

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
Within AutoML the experiment was set to time out in 20 minutes, the maximum number of compute nodes was set to 3, contingent upon the compute cluster having 4 nodes provisioned.The primary metric for training was set to accuracy. Featurization settings were also set to automatic.


### Results
The best model that was selected from AutoML was a Random Forest Classifier. The minimum sample split parameter value was 0.338, and the number of esitmators was set to 200. There was no max depth and the splitting criterion was the gini coefficient. An exhaustive description of the model is below. It achieved an accuracy of 82%.

In terms of AutoML, improvments could have been made by changing the stopping criteria and allowing for more iterations of training runs, as this would generate a greater volume of model runs. 

```
Pipeline(memory=None,
         steps=[('datatransformer',
                 DataTransformer(enable_dnn=False, enable_feature_sweeping=True, feature_sweeping_config={}, feature_sweeping_timeout=86400, featurization_config=None, force_text_dnn=False, is_cross_validation=True, is_onnx_compatible=False, observer=None, task='classification', working_dir='/mnt/batch/tasks/shared/LS_root/mount...
), random_state=None, reg_alpha=0.5789473684210527, reg_lambda=0.42105263157894735, subsample=1))], verbose=False)), ('11', Pipeline(memory=None, steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('randomforestclassifier', RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced', criterion='gini', max_depth=None, max_features='sqrt', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=0.01, min_samples_split=0.33789473684210525, min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))], verbose=False))], flatten_transform=None, weights=[0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285, 0.14285714285714285]))],
         verbose=False)
```

![Screen Shot 2022-02-24 at 9 42 52 PM](https://user-images.githubusercontent.com/87383001/155643986-25906b5f-9175-4a03-a96e-0d95afa5483a.png)

<img width="1049" alt="Screen Shot 2022-02-21 at 8 59 36 PM" src="https://user-images.githubusercontent.com/87383001/155050482-e67f53f9-94cc-4512-a448-044cf7ee28c7.png">


## Hyperparameter Tuning
A Logistic regression was chosen for this experiment. This model was selected because the task called for binary classification, and Logistic regression is apt for this use case. The hyperparameters chosen for training included C, to prevent overfitting, and max iterations to make sure that the regression fit converges. C was randomly sampled from a uniform distribution bounded by 0.005 and 1. The max iteration parameter was randomly sampled from a discrete choice distribution comprised of a set of 100, 200, 500, and 1000. The early termination policy was set to bandit, with evaluation intervals beginning after 2 runs, evaluations happening every 2 runs, and a slack factor of 0.2.


### Results
The model that was generated from hyperdrive achieved a maximum of 0.725 accuracy. The optimal parameters were C = 0.814 and max_iter = 500. The model could be improved in the future by altering the bandit policy to allow for a more extensive set of runs. 

![Screen Shot 2022-02-24 at 10 25 44 PM](https://user-images.githubusercontent.com/87383001/155648389-57333074-17e4-4417-8fac-4ee2cb7e47ab.png)
![Screen Shot 2022-02-24 at 10 29 44 PM](https://user-images.githubusercontent.com/87383001/155648440-82229ebb-eba1-4e33-a162-3cd4cbbb42d7.png)
![Screen Shot 2022-02-24 at 10 25 55 PM](https://user-images.githubusercontent.com/87383001/155648408-c694282d-976d-45fd-8d17-229802156453.png)



## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
The best model from the AutoML run was chosen for deployment. This was done via instantiation of an Azure Container instance as a web service. The inference configuration was assigned from the environment that was used during training of the deployed model. The container instance was created with 0.5 cpu cores and 1 gigabyte of memory, and was enabled for authorization. The endpoint can be queried to recieve a classification. The json input is defined like below:
```python
uri = web_service.scoring_uri
api_key = 'EvWeVi2rnmYBE7S0XZlEwF6HadSX6zMA'
headers = {"Content-Type": "application/json",  'Authorization':('Bearer '+ api_key)}
data = {
    "fixed acidity":7.3,	
    "volatile acidity": 0.9,	
    "citric acid": 0.1,	
    "residual sugar": 1.9,	
    "chlorides": 0.076,	
    "free sulfur dioxide": 11.0, 	
    "total sulfur dioxide": 34.0,	
    "density": 0.9978,	
    "pH": 3.49,	
    "sulphates": 0.50,	
    "alcohol": 9.4
}
data = json.dumps(data)
response = requests.post(uri, data=data, headers=headers)
print(response.json())
```
The entry script echo_score.py is used to produce a response:
```python
import json
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    # retrieve model by name
    path = Model.get_model_path('automl-model')
    model = joblib.load(path)

def run(input_data):
    data = json.loads(input_data)
    data_pred = pd.DataFrame(data, index=[0])
    # make prediction 
    y = model.predict(data_pred)
    return json.dumps({"classification": int(y[0])})


```

## Screen Recording
https://youtu.be/zGEW0pHjPqM
