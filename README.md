# Neural_Network_Charity_Analysis


![artificial_neural_networks](https://user-images.githubusercontent.com/85518330/146293771-d290c9dc-32c6-4747-a423-77de34fe3621.jpg)




## Overview of the analysis

In this Module we will be helping Beks & Andy develop a model that will accurately predict sound investments for **Alphabet Soup foundation**.
Alphabet Soup is a philanthropic organization dedicated to help organizations that protect the environment, improve the well being of people and unify the world. As per the data available for analysis only 53% of all the projects have a successful outcome. So Andy wants Beks to help him develop a way to know which  organizations are worth donating to using a mathematical, data driven solution that will make accurate predictions. Beks feels that this problem will be best solved using a **Deep Learning Neural Network** 


## Resources

Data Source: charity_data.csv

Software: Python 3.7.7, Anaconda Navigator 1.9.12, Conda 4.8.4, Jupyter Notebook 6.0.3

Libraries - TensorFlow, scikit, pandas 


## Results

### Data Preprocessing

* We removed the **"EIN" and "NAME"** columns as they did not offer any relevant data that could help the model perform better. 

* The **APPLICATION_TYPE** and **CLASSIFICATION** columns were binned 

* We used **OneHotEncoder** to transform all columns of categorical data to numerical data. Since machine learning algorithms cannot work with categorical data directly.

* We merged the OneHotEncoded dataframe with the original dataframe and dropped the original categorical columns.

* We split our preprocessed data into our features and target arrays - The target variable is **"IS_SUCCESSFUL"**  & The remaining columns became the features for the model

* Split the preprocessed data into a training and testing dataset

* Used StandardScalar instance to standardize the data to have a mean of zero and Std deviation of 1 

 
### Compiling, Training, and Evaluating the Model

* Next we define the deep neural net model, 

* The number of input features - 43 

* The hidden layers - 2 

* The number of nodes in layer 1 = 80 (Norm is to have nodes between 2-3 times that of the input layer)

* The number of nodes in layer 2 = 30

* Activation function for hidden layers - ReLU

* Activation function for output layer - Sigmoid 

* We trained the model using 100 epochs

* Training data  Results  -  loss: 0.5360 - accuracy: 0.7404

* Test data Results were as below

<img width="330" alt="AlphabetSoup" src="https://user-images.githubusercontent.com/85518330/138931148-d1c3bb96-3744-4acd-8fa2-e1ef345f6db0.png">

* Our model achieved a target predictive accuracy lower than 75%, so in the next few attempts we aimed to increase the same to 75%

#### Optimization Attempt - 1 

In this attempt we reduced the APPLICATION_TYPE bins from 9 to 6 & created bins for ASK_AMT.  Our intention was to smooth data and to handle noisy data better. In this method, the data was distributed into a number of buckets or bins. We used two hidden layers with 80 and 30 nodes respectively and used Activation function Relu for the hidden layers and sigmoid for the output layer. We used 100 epochs. The performance of this model is as under

<img width="346" alt="Optimization Attempt -  1" src="https://user-images.githubusercontent.com/85518330/138936691-ba173845-fa18-4f83-8ed3-5057bad05305.png">

#### Optimization Attempt - 2 

In this attempt we added more neurons to our hidden layers.  Hidden layer 1 increased to 138 and hodden layer 2 to 60 nodes respectively we used Activation function Relu for the hidden layers and sigmoid for the output layer. We used 100 epochs. The performance of this model is as under

<img width="323" alt="Optimization Attempt2" src="https://user-images.githubusercontent.com/85518330/138937565-79d4eb96-6171-40ab-8e3a-80a980bebeba.png">


#### Optimization Attempt - 3 

In this attempt we added a third hidden layer. The performance of this model is as under

<img width="372" alt="Optimization Attempt3" src="https://user-images.githubusercontent.com/85518330/138938246-f14f8682-81e0-4d14-8eaa-709b71dd479b.png">


#### Optimization Attempt - 4

In this attempt we increased epochs from 100 to 200. The performance of this model is as under

<img width="338" alt="Optimization Attempt 4" src="https://user-images.githubusercontent.com/85518330/138939325-8f38edd8-d8e4-48eb-9991-09367b34a64e.png">


#### Optimization Attempt - 5

In this attempt used Random Forest Classifier instead just to check the impact on the results. The performance of this model is as under

<img width="474" alt="Random_Forest" src="https://user-images.githubusercontent.com/85518330/138939582-7c3a85ad-c049-4cc1-90b2-e3716f179a58.png">


## Optimization Summary
After four attempts and multiple levels of optimization, we were unable to get the model with a 75% accuracy rating. The models consistently performed at an accuracy of 72% through all our optimization attempts. We havent tried changing the optimizer or the loss in any of our optimizations. That maybe something we should look into in greater detail
