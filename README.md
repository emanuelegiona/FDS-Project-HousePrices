# FDS-Project-HousePrices
Project for Fundamentals of Data Science 2018/2019, from the MSc in Computer Science.

Forked from [luigiberducci][project], group composed by: [luigiberducci][luigi], [angelodimambro][angelo], and I.

**Kaggle Score:** 0.11440

## Feature engineering overview

- 3 new features introduced: total number of bathrooms, number of garage cars multiplied by garage area, total square feet
- removal of multicollinear features
- automatic removal of features receiving [caret importance score][varImp] equal to 0 when considering a Lasso regression model, until the RMSE value of such model didn't decrease any further

## Models overview

### Simple models

- Lasso regression model
- Ridge regression model
- eXtreme Gradient Boosting model
- Support Vector Machines

### More complex models

- Ensemble model (average)
- Stacked regression model (both variants A and B)

#### Ensemble model

Our ensemble model performs a weighted average of predictions produced of a set of simple models, using the following weights and models:

<table>
  <tr>
    <td><b>Model</b></td>
    <td><b>Weight</b></td>
  </tr>
  
  <tr>
    <td>Lasso</td>
    <td>0.5</td>
  </tr>
  
  <tr>
    <td>Ridge</td>
    <td>0.5</td>
  </tr>
  
  <tr>
    <td>XGB</td>
    <td>3.5</td>
  </tr>
  
  <tr>
    <td>SVM</td>
    <td>5</td>
  </tr>
</table>

Such weights have been optimized via 10-fold CV, minimizing the average RMSE and weights themselves.

#### Stacked regression model

A set of simple models' predictions is used to train a meta-model.

Variants:

- Variant A: meta-model trained on the average of the predictions produced during the simple models' k-fold trainings
- Variant B: meta-model trained on predictions produced by new instances of the simple models, those being trained on the whole training set

Our stacked regression model uses the following recipe:

<table>
  <tr>
    <td><b>Simple models</b></td>
    <td><b>Meta-model</b></td>
  </tr>
  
  <tr>
    <td>Lasso</td>
    <td rowspan="4">Specific XGB</td>
  </tr>
  
  <tr>
    <td>Ridge</td>
  </tr>
  
  <tr>
    <td>XGB</td>
  </tr>
  
  <tr>
    <td>SVM</td>
  </tr>
</table>

## Final predictions

Our final predictions are computed in the following way:

`predictions = ( 2 * ensemble + xgb + svm + stacked_variantA + stacked_variantB ) / 6`

[project]: https://github.com/luigiberducci/FDS-Project-HousePrices
[luigi]: https://github.com/luigiberducci
[angelo]: https://github.com/angelodimambro
[varImp]: https://www.rdocumentation.org/packages/caret/versions/6.0-81/topics/varImp
