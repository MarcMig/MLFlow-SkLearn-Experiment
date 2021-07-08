from sklearn.datasets import load_boston
from sklearn import linear_model
import mlflow
import argparse

def get_flags_passed_in_from_terminal():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default='LinearRegression')
	args = parser.parse_args()
	return args

args = get_flags_passed_in_from_terminal()
print(args.model)

#mlflow.set_tracking_uri("file:/C:\Users\marcm\mlruns")

model = getattr(linear_model, args.model)()

print(model)
print(type(model))

X, y = load_boston(return_X_y = True)

with mlflow.start_run():
    # model = models[args.model]
    model.fit(X, y)
    
    score = model.score(X, y)
    mlflow.log_metric('R-Squared', score)

    mlflow.sklearn.log_model(
		sk_model=model,
		artifact_path="sklearn-model"
	)