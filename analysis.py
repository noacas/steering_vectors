from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import collections
import pandas as pd

from consts import MIN_LEN
from dot_act import get_dot_act, get_mean_dot_prod
from model import ModelBundle
from utils import dict_subtraction


def compute_component_prediction_r2(dot_prod_dict_train, dot_prod_dict_test, residual_stream_component_name = 'blocks.10.hook_resid_pre'):
    # Expecting that dot_prod_dict is a nested dict of example: {component: dot product} pairs
    component_names = dot_prod_dict_train[0].keys()
    total_dot_products_train = np.array([dot_prod_dict_train[example][residual_stream_component_name] for example in range(len(dot_prod_dict_train))])
    total_dot_products_test = np.array([dot_prod_dict_test[example][residual_stream_component_name] for example in range(len(dot_prod_dict_test))])
    r2s = dict()
    for component_name in component_names:
        component_dot_products_train = np.array([dot_prod_dict_train[example][component_name] for example in range(len(dot_prod_dict_train))])
        component_dot_products_test = np.array([dot_prod_dict_test[example][component_name] for example in range(len(dot_prod_dict_test))])

        lr = LinearRegression()
        lr.fit(component_dot_products_train.reshape(-1, 1), total_dot_products_train)
        pred = lr.predict(component_dot_products_test.reshape(-1, 1))

        r2s[component_name] = r2_score(total_dot_products_test, pred)
    return r2s


def compare_component_prediction_r2(model_bundle: ModelBundle):
    train_dict = collections.defaultdict(dict)
    test_dict = collections.defaultdict(dict)

    for pos in range(-1, -MIN_LEN - 1, -1):
        print("pos = ", pos)
        subset_len = len(model_bundle.harmful_inst_train) # There is a great imbalance in dataset size

        model = model_bundle.model
        refusal_direction = model_bundle.refusal_direction

        harmless_outputs_train = get_dot_act(model, model_bundle.harmless_inst_train[:subset_len], pos, refusal_direction)
        harmful_outputs_train = get_dot_act(model, model_bundle.harmful_inst_train[:subset_len], pos, refusal_direction)
        harmless_outputs_test = get_dot_act(model, model_bundle.harmless_inst_test[:subset_len], pos, refusal_direction)
        harmful_outputs_test = get_dot_act(model, model_bundle.harmful_inst_test[:subset_len], pos, refusal_direction)

        # Get per component r2 score
        harmless_r2s = compute_component_prediction_r2(harmless_outputs_train, harmless_outputs_test)
        harmful_r2s = compute_component_prediction_r2(harmful_outputs_train, harmful_outputs_test)

        diff_in_means_train = get_mean_dot_prod(harmful_outputs_train)
        diff_in_means_train = dict_subtraction(diff_in_means_train, get_mean_dot_prod(harmless_outputs_train))

        diff_in_means_test = get_mean_dot_prod(harmful_outputs_test)
        diff_in_means_test = dict_subtraction(diff_in_means_test, get_mean_dot_prod(harmless_outputs_test))

        # Sort the diff in means by train
        diff_in_means_train_list = list(diff_in_means_train.items())
        diff_in_means_train_list.sort(key=lambda x: abs(x[1]), reverse=True)

        for component_name, _ in diff_in_means_train_list:
            diff_train = diff_in_means_train[component_name]#.item()
            diff_test = diff_in_means_test[component_name]#.item()

            train_dict[pos][component_name] = diff_train
            test_dict[pos][component_name] = diff_test
            print(f"{component_name}: Diff in means - Train: {diff_train:.4f} Test: {diff_test:.4f}")
            print(f"{component_name}: R2 - Harmless: {harmless_r2s[component_name]:.4f} Harmful: {harmful_r2s[component_name]:.4f}")
            print()

        # del harmless_outputs_train
        # del harmless_outputs_test
        # del harmful_outputs_train
        # del harmful_outputs_test
        # gc.collect()

    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)
    train_df.to_csv('train_df.csv')
    test_df.to_csv('test_df.csv')