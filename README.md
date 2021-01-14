# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

In this project, a classifier was trained to predict whether a client with certain characteristics will subscribe to a term deposit or not with the bank. Two options are compared, on the one hand the HyperDrive option to optimize the best hyperparameters of the logistic regression of the scikit-learn framework, and the second option the Azure AutoML using the Azure Machine Learning Service SDK.

The data set is related to marketing campaigns of a banking institution. The campaigns were based on phone calls. As mentioned above, the objective is to predict whether a term bank deposit was signed or not.

The idea is to predict this variable based on characteristics such as customer data such as sociodemographic information, behavior within the financial institution, some macroeconomic variables and information at the time and after the campaign.

In the data preprocessing, some steps were done to leave the data in a clean way for model training:

* Null values removed.
* The variables referred to dates (month, days of the week) were transformed into numerical variables.
* The categorical variables were coded as dummy variables (ohe).

After this preprocessing step, 12 iterations were experimented to find the best hyperparameters of the logistic regression using HyperDrive. The best performance was an Accuracy of 91.13%, while using AutoML a performance in Accuracy of 91.71% was obtained.

## Scikit-learn Pipeline

In the Sklearn pipeline, we used the logistic regression model for classification with the adjustment of the hyperparameters using Hyperdrive. The hyperparameters used are: C (Inverse of the regularization factor) and max_iter (maximum number of iterations). In order to adjust the hyperparameters, we have used the main metric Accuracy in order to maximize it.

The RandomParameterSampling was used which randomly selects the values of the parameters in some range defined by the system. For our problem we define a maximum of 12 iterations and it is a way to optimize the hyperparameters with less computational cost than looking at a complete grid, for example.

I used the Bandit policy, where this terminates runs where the main metric is not within the slack factor compared to the best performance run. This ignores runs that will not give the best results and will help decrease experimentation time.

## AutoML

Using AutoML, the task was defined as "classification" and the main metric to optimize as "Accuracy" using the default iterations, without allowing the process to exceed 30 minutes. The best performance was obtained with the VotingEnsemble algorithm with an Accuracy of 91.71%.

AutoML allows ensemble models, as is the case with the VotingEnsemble. First, the assembly methods use different algorithms, even those algorithms can be defined as weak learners, however, when they work together they can obtain very good results and deal with problems such as the trade-off between bias and variance.

The VotingEnsemble predicts based on the weighted average of the predicted probabilities for each class in the case of a classification problem like the one we have. Some optimal hyperparameters of the VotingClassifier that we found:

* **reg_lambda**: 1.35. It is a hyperparameter that helps us with the regularization to avoid overfitting, one expects that the larger this value the stronger the regularization.
* **scale_pos_weight**: 1. It helps us to give a weight to the classes, it is a hyperparameter widely used to treat problems of unbalanced classes.

## Pipeline comparison

Our experiments give us the following results:

* Accuracy: 91.13% using Hyperdrive.
* Accuracy: 91.71% using AutoML.

We see that Accuracy is much better using AutoML which makes sense given that many more models are used with different transformations, however, using logistic regression we do not seek the optimization of a very important hyperparameter for class unbalance, and we see that the The model is very good at predicting the "no" in the test data and not so much the "yes".

## Future work

As mentioned above, it is necessary to check if there is a class imbalance that AutoML did, in order to experiment with other metrics such as recall to reduce the FN that we are obtaining with both (HyperDrive and AutoML) or even the F1 Score looking for a balance between FP and FN.
