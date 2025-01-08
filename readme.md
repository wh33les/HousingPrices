# House Prices - Advanced Regression Techniques
## Predict sales prices and practice feature engineering, RFs, and gradient boosting

[Kaggle competition.](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)  Original data is in the `download` folder.

(1) Data exploration.  In `data_exploration.ipynb`.  Plots of distributions of numerical vs. categorical data, as well as the training data features against the `SalePrice`.  Some prepreprocessing like imputing.  Prepreprocessed data exported to the `raw_data` folder.

(2) Preprocessing.  In the `preprocessing` folder.  Data is split for k-fold cross-validation.  The random seed is stored in `rs_log.xt` with a timestamp.  More imputing, according to the train splits.  One-hot encoding.  MCA on the categorical features and kept 80% explained variance.

(3) Model selection.  Based on the plots in `data_exploration.ipynb` the relationship between most of the features and the target does not appear to be linear.  The first thing I checked was the residuals of a linear regression model.  It seems to be within 100,000 of the true value, except for small and large values of `SalePrice`.  Based on that, and the data exploration plots, I decided a good baseline would be polynomial regression of degree 2, with elastic net.

The other models I tested were `RandomForestRegressor`, `AdaBoostRegressor`, `XGBRegressor`, and a custom neural network I built using `torch`.  For the neural network I used two linear layers with a leaky ReLu activation function and a hidden layer size of 32.  Adam optimizer with a learning rate of 0.0001.