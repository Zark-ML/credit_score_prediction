import optuna
from sklearn.metrics import mean_squared_error

def optuna_hyperparameter_tuning(regression_model, params, **kwargs):
    def objective(trial):
        # Sample hyperparameters
        param_dict = {}
        for param_name, param_range in params.items():
            if isinstance(param_range[0], float):  # If the parameter is continuous
                param_value = trial.suggest_uniform(param_name, param_range[0], param_range[1])
            else:  # If the parameter is categorical
                param_value = trial.suggest_categorical(param_name, param_range)
            param_dict[param_name] = param_value
        
        # Initialize model with sampled hyperparameters
        model = regression_model(**param_dict)
        model.fit(kwargs['X_train'], kwargs['y_train'])
        y_pred = model.predict(kwargs['X_test'])
        
        # Calculate mean squared error
        mse = mean_squared_error(kwargs['y_test'], y_pred)
        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # Train final model with best parameters
    print(best_params, "Best")
    best_model = regression_model(**best_params, random_state=42)
    best_model.fit(kwargs['X_train'], kwargs['y_train'])

    # Evaluate best model
    y_pred = best_model.predict(kwargs['X_test'])
    best_mse = mean_squared_error(kwargs['y_test'], y_pred)
    print("Best Mean Squared Error:", best_mse)

# Example usage:
# Assuming regression_model is your model class (e.g., LinearRegression, RandomForestRegressor, etc.)
# params = {'param_name1': (min_value1, max_value1), 'param_name2': (min_value2, max_value2), ...}
# kwargs = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
# optuna_hyperparameter_tuning(regression_model, params, **kwargs)
