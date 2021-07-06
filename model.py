from sklearn.datasets import load_boston
from sklearn import linear_model
import mlflow
import argparse

def get_flags_passed_in_from_terminal():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model')
	args = parser.parse_args()
	return args

args = get_flags_passed_in_from_terminal()
print(args.model)

models = {
    'Linear': linear_model.LinearRegression(),
    'Ridge': linear_model.Ridge(),
    'Lasso': linear_model.Lasso(),
    'ElasticNet': linear_model.ElasticNet(),
    'Tweedee': linear_model.TweedieRegressor()
}


X, y = load_boston(return_X_y = True)

with mlflow.start_run():
    model = models[args.model]
    model.fit(X, y)
    
    score = model.score(X, y)
    mlflow.log_metric('R-Squared', score)