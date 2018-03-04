# method section for ALL-10 paper

## Pre-processing

imputance/corrupt-data removal 

## Cohort bias removal by cohort-wise averaging

## Data reduction


### LDA / QDA / PCA 

mathematical description.

### variance filter

FDR ANOVA and Mann-Whitney U with maximum p-value of $0.05$

### Sample reduction

* Outlier detection: to reduce overfitting we can apply a dimension reduction technique
such as the isolation forest method. 
* healthy tissue removal: more specifically, we might identify tissue samples that are in 
fact non-cancerous. 


## Classification algorithm

### Tree methods
* 	XGBoost
* 	Extratrees
* 	RandomForest
*	LightGBM
*	Adaboost
*	CART

### Neural networks
*	DNN (MLP)
*	CNN

### Kernel methods
*	SVM
*	RVM

### Linear methods
*	Logistic Regression
*	Linear SVM

## Bagging

Given the wide array of classification methods we can apply 
another classification algorithm to combine the previous models in one model.

## Post-processing

*	Importance vectors, for the tree methods
*	Weight vectors, only for the linear methods

## Caveat

The neural network methods are very accurate but provide little obvious insight in the contributions
of individual genomes.