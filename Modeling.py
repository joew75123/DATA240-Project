from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def LinearRegression_cus(X_train, X_val, y_train, y_val):
    # assuming X_train and y_train are the training set and target variable
    # fit the linear regression model
    print("Linear Regression:")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # make predictions on the validation set
    y_pred = lr.predict(X_val)

    # calculate the mean squared error
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    print("Validation Mean Squared Error:", mse)
    print("Validation Root Mean Squared Error:", rmse)
    print("Validation Mean Absolute Error:", mae)
    return mse, lr


def RandomForest(X_train, X_val, y_train, y_val):
    print("Random Forest:")
    
    # Define the hyperparameters to search over
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
    }

    # Create the random forest regressor
    rf = RandomForestRegressor(random_state=42)

    # Create the grid search object
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the grid search object to the training data
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Best Mean Squared Error:", -grid_search.best_score_)

    # Use the best estimator to make predictions on the validation data
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_val)

    # Calculate the mean squared error on the validation data
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    print("Validation Mean Squared Error:", mse)
    print("Validation Root Mean Squared Error:", rmse)
    print("Validation Mean Absolute Error:", mae)
    return mse, best_rf


def GradientBoosting(X_train, X_val, y_train, y_val):
    print("Gradient Boosting:")

    # Define the Gradient Boosting model
    gb = GradientBoostingRegressor()

    # Define hyperparameters to search over
    param_grid = {
        'n_estimators': [50, 100, 200, 400],
        'learning_rate': [0.01, 0.1, 0.5],
    }

    # Perform a grid search over hyperparameters
    gb_cv = GridSearchCV(gb, param_grid, scoring='neg_mean_squared_error', cv=5)
    gb_cv.fit(X_train, y_train)

    # Print the best hyperparameters and corresponding mean squared error
    print("Best Parameters:", gb_cv.best_params_)
    print("Best Mean Squared Error:", -gb_cv.best_score_)

    # Predict on the validation set using the best model
    best_gb = gb_cv.best_estimator_
    y_pred = best_gb.predict(X_val)

    # Calculate and print the mean squared error on the validation set
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    print("Validation Mean Squared Error:", mse)
    print("Validation Root Mean Squared Error:", rmse)
    print("Validation Mean Absolute Error:", mae)
    return mse, best_gb


def KNN(X_train, X_val, y_train, y_val):
    print("KNN:")

    # create KNN regression object
    knn = KNeighborsRegressor()

    # specify hyperparameters to tune
    param_grid = {'n_neighbors': range(5,20,2), 'weights': ['uniform', 'distance']}

    # use grid search cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # print the best hyperparameters and corresponding mean squared error
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Best MSE: ", -grid_search.best_score_)

    # fit the model with the best hyperparameters on the training data
    best_knn = grid_search.best_estimator_
    best_knn.fit(X_train, y_train)

    # predict on the validation data
    y_pred = best_knn.predict(X_val)

    # evaluate the model on the validation data
    mse = mean_squared_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    print("Validation Mean Squared Error:", mse)
    print("Validation Root Mean Squared Error:", rmse)
    print("Validation Mean Absolute Error:", mae)
    return mse, best_knn


class Modeling:
    def __init__(self):
        pass

    def Evaluate(self, X_train, X_test, y_train, y_test, X_val, y_val):
        print("Beginning Evaluation:")
        mse_lr, best_lr = LinearRegression_cus(X_train, X_val, y_train,
                                               y_val)
        mse_rf, best_rf = RandomForest(X_train, X_val, y_train, y_val)
        mse_gb, best_gb = GradientBoosting(X_train, X_val, y_train, y_val)
        mse_knn, best_knn = KNN(X_train, X_val, y_train, y_val)
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

        print(
            'The best model is: ', best_model,
            ', because it has the lowest validation MSE. Final  model performance on the test set are as follows: '
        )

        print('MSE:', mse)
        print('RMSE:', rmse)
        print('MAE:', mae)
        
        # Visualize the performance using a scatter plot
        plt.scatter(y_test, y_pred)
        plt.plot([0, 10], [0, 10], 'r--')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual Prices vs Predicted Prices')
        plt.savefig(f"Figure/Evaluation/scatter_plot.png")
        plt.close()
        
        residuals = y_test - y_pred

        # Plot a histogram of the residuals
        plt.hist(residuals, bins=20)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        plt.savefig(f"Figure/Evaluation/Residuals.png")
        plt.close()