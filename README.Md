# DD2421 - Machine Learning Repository

Made by FLANDRE Corentin & MADEC Steven

See also *DD2434 - Machine Learning, Advanced Course Repository* corresponding to the course of the period 2

This repository contains all works labs from us :

**1. Lab1 - Decision Trees**


**2. Support Vector Machines**

**3. Bayes Classifier & Boosting**

## Lab1 - Decision Trees

> Corresponding to the repository *LAB1-dectrees*

> Importants files for this laboratory are : *LAB1-report-FLANDRE-MADEC-v2.pdf* (which corresponds to the report) and *lab1_FLANDRE_MADEC.py* (which corresponds to the personal code)

Basics of Machine Learning with scientific approach including understanding of what's a dataset and the difference between learning and testing set.  

Concepts of decision trees with calculation methods during construction : 
- Entropy
- Gain information

Improving decision tree methods with pruning method and a parameter of fractionning to a training and validation sets with the real validation set approach view


## Lab2 - Support Vector Machines

> Corresponding to the repository *LAB2-svm*

> Importants files for this laboratory are : *LAB2-report-FLANDRE-v1.pdf* (which corresponds to the report) and *LAB2-dev.py* (which corresponds to the personal code)

Implementation of a SVM model for classification

Concepts of SVM classification:
- Decision boundaries 
- Support Vector and Margin
- Dual Formulation Theory

Improving the SVM model:
- Kernel functions (linear function, polynomial function, radius basis function(RBF))
- Slack variables

Paying attention to :

1. Reaction of the model/optimizer when it’s not able to find a solution

2. Implementation of non-linear kernels and impact on classification 

3. Influence of parameters of non-linear kernels in terms of the bias-variance trade-off 

4. Exploring slack parameter C in large/small values (increase/decrease value)

5. Balancing between opt for more slack or for more complex model

## Lab3 - Bayes Classifier & Boosting

> Corresponding to the repository *LAB3-bayes+boosting*

> Importants files for this laboratory are : *LAB3-report-v3.pdf* (which corresponds to the report) and *LAB3-report.ipynb* (which corresponds to the jupyter notebook)

Implementation of a Bayes Classifier and a Boosting classifier function

Usage of a Decision Classifier and its boosting version

This lab paying attention to :

1. Bayes' theorem

2. Probabilistic reasoning (especially prior, likehood, evidence and posterior probabilities)

3. Boosting effect (implentation of AdaBoost algorithm)