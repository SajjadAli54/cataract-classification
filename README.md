# Methodology

## Data Visualization
	5 techniques
- BGR : Blue Green Red
- RGB : Red Green Blue
- Graye Level
- Adaptive Threshold Gray Level
- Gray Level Co occurrence Matrix (GLCM) features 
	'dissimilarity', 'contrast', 'homogeneity', 'energy', 'ASM', 'correlation'

## Data Augmentation
	Augmented by Gaussian Filter
	30% data of each class is augmented and then saved.
	30% increase in data (600 to 750+)

## Feature Extraction using GLCM
	'dissimilarity', 'contrast', 'homogeneity', 'energy', 'ASM', 'correlation'

## Models (Five models)
	SVM, KNN, Logistic Regression, Random Forest, Naive Bayes

## Selection models technique
	Learning curve
	Validation Curve

	Both use Stratified KFold cross validation by default

## Metrics
	Accuracy, Precision, Recall, F1 Score, and AUC ROC score

## Conclusion
	KNN performs better most of the time. It gave a better accuracy of almost 64% with auc roc score of 
	0.90 as  compare to other five models. Analyzed overfitting, underfitting, variance, and bias too.