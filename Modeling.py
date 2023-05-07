from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def LinearRegression_cus(X_train, X_test, y_train, y_test):
    # assuming X_train and y_train are the training set and target variable
    # fit the linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # make predictions on the test set
    y_pred = lr.predict(X_test)

    # calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Test Mean Squared Error for LinearRegression:", mse)
    return mse, lr


def RandomForest(X_train, X_test, y_train, y_test):
    # Define the hyperparameters to search over
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
    }

    # Create the random forest regressor
    rf = RandomForestRegressor(random_state=42)

    # Create the grid search object
    grid_search = GridSearchCV(rf,
                               param_grid,
                               cv=5,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1)

    # Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    #print("Best hyperparameters: ", grid_search.best_params_)
    #print("Best Mean Squared Error:", -grid_search.best_score_)

    # Use the best estimator to make predictions on the test data
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    # Calculate the mean squared error on the test data
    mse = mean_squared_error(y_test, y_pred)
    print("Test Mean Squared Error for RandomForest: ", mse)
    return mse, best_rf


def GradientBoosting(X_train, X_test, y_train, y_test):
    # Define the Gradient Boosting model
    gb = GradientBoostingRegressor()

    # Define hyperparameters to search over
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.5],
    }

    # Perform a grid search over hyperparameters
    gb_cv = GridSearchCV(gb,
                         param_grid,
                         scoring='neg_mean_squared_error',
                         cv=5)
    gb_cv.fit(X_train, y_train)

    # Print the best hyperparameters and corresponding mean squared error
    #print("Best Parameters:", gb_cv.best_params_)
    #print("Best Mean Squared Error:", -gb_cv.best_score_)

    # Use the best estimator to make predictions on the test data
    best_gb = gb_cv.best_estimator_
    y_pred = best_gb.predict(X_test)

    # Calculate and print the mean squared error on the test set
    mse = mean_squared_error(y_test, y_pred)
    print("Test Mean Squared Error for GradientBoosting:", mse)
    return mse, best_gb


def KNN(X_train, X_test, y_train, y_test):
    # create KNN regression object
    knn = KNeighborsRegressor()

    # specify hyperparameters to tune
    param_grid = {
        'n_neighbors': range(5, 20, 2),
        'weights': ['uniform', 'distance']
    }

    # use grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(knn,
                               param_grid,
                               cv=5,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # print the best hyperparameters and corresponding mean squared error
    #print("Best hyperparameters: ", grid_search.best_params_)
    #print("Best MSE: ", -grid_search.best_score_)

    # fit the model with the best hyperparameters on the training data
    best_knn = grid_search.best_estimator_
    best_knn.fit(X_train, y_train)

    # predict on the test data
    y_pred = best_knn.predict(X_test)

    # evaluate the model on the test data
    mse = mean_squared_error(y_test, y_pred)
    print("Test Mean Squared Error for KNN: ", mse)
    return mse, best_knn


class Modeling:
    def __init__(self):
        pass

    def Evaluate(self, X_train, X_test, y_train, y_test):
        mse_lr, best_lr = LinearRegression_cus(X_train, X_test, y_train,
                                               y_test)
        mse_rf, best_rf = RandomForest(X_train, X_test, y_train, y_test)
        mse_gb, best_gb = GradientBoosting(X_train, X_test, y_train, y_test)
        mse_knn, best_knn = KNN(X_train, X_test, y_train, y_test)
        mse_dic = {
            best_lr: mse_lr,
            best_rf: mse_rf,
            best_gb: mse_gb,
            best_knn: mse_knn
        }
        best_model = min(mse_dic, key=mse_dic.get)
        # y_pred
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(
            'The best model is: ', best_model,
            ', because it has the lowest test MSE. Final  model performance on the test set are as follows: '
        )

        print('MSE:', mse)
        print('RMSE:', rmse)
        print('MAE:', mae)
        print('R-squared:', r2)