# Real-time-deep-emotion-detection
The aim of the project is to feed a deep CNN with webcam inputs and evaluate if an expression is happy or not.
The model I want to try is an ensemble forecasting system which will gather the predictions from 7 different deep CNNs trained to distinguish each a different expression from happiness.
The synthesis can be done through different voting rules, in general i found to be working nicely the Majority Voting Rule: if most of the classifiers detect happiness then the outcome shall be happiness. <br>

The confidence of the accuracy can be exressed in terms of how mnay voters are in favor for *'happiness'*:<br>

$$ confidence = \sum_{k=1}^{N} output_k /N$$

# Workflow:
## 1. build and train the model 
I trained the individual models on the Face Expression Recognition Dataset publicly available on Kaggle 
(https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset). One inconvinience is that the size of the training elements is realitvely small; if you are
interested in the details I suggest you to check my commented notebook the notebook 1 where the pipeline is 
explained step by step including a trick to super-sample up to 256x256 without any explicit command.

In notebook 2 the model architecture is not discussed however, an outline of the ensemble forecasting is given. 

## 2. save the models and make the ensemble
This part refers to ModelDeployment.py where the ensemble voting function is defined. <br>
The ensemble voting function takes as input a parameter to explicit the voting rule and the models that contribute to the outcome

## 3. deploy the models
Open the webcam, send each frame through the ensemble function, display on screen

## Bonus: Color encoding
If you look sad the words should be blue. If you look happy, words sould be red ;)

# RESULTS
The model is not perfect. There are many things to improve, from the dataset to the ensemble forecasting itself. Under this aspect, the voting system should be able to weight more confident votes from specific voters or weighting less the most uncertain ones (i.e. surprise ~ happy so 'not happy' might actually be 'happy') 
