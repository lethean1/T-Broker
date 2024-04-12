import json
import pandas as pd
import statsmodels.api as sm

train_data = pd.read_csv("train_data_gpulet.csv")

train_xs = train_data[["l2m1","l2m2","memm1","memm2"]]
train_ys = train_data["interference_factor"]

train_xs = sm.add_constant(train_xs)

linear_regression_model = sm.OLS(train_ys, train_xs)
fit_result = linear_regression_model.fit()
print(fit_result.summary())

params = fit_result.params

test_data = pd.read_csv("test_data_gpulet.csv")

test_truth = test_data["interference_factor"]

test_l2m1 = test_data['l2m1']
test_l2m2 = test_data['l2m2']
test_memm1 = test_data['memm1']
test_memm2 = test_data['memm2']

pred = params[0] + test_l2m1 * params[1] + test_l2m2 * params[2] + test_memm1 * params[3] + test_memm2 * params[4]

abs_diff = abs(pred - test_truth)
rel_diff = abs_diff / test_truth

print(rel_diff.mean())

# export data
with open("gpulet_test_accuracy.json", "w") as f:
    data = []
    for i in range(len(rel_diff)):
        data.append(float(rel_diff[i]))
    json.dump({"data": data}, f, indent=2)