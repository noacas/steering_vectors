import itertools
import os

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import collections
import pandas as pd
import torch

from consts import MIN_LEN
from dot_act import get_dot_act, get_mean_dot_prod, get_act
from model import ModelBundle
from utils import dict_subtraction


def compute_component_prediction_r2(dot_prod_dict_train, dot_prod_dict_test, dot_prod_dict_train_norm = None, dot_prod_dict_test_norm = None, residual_stream_component_name = 'blocks.10.hook_resid_pre'):
    # Expecting that dot_prod_dict is a nested dict of example: {component: dot product} pairs
    assert dot_prod_dict_train_norm is None or dot_prod_dict_train_norm is not None, "If you provide dot_prod_dict_train_norm, you must also provide dot_prod_dict_test_norm"
    normalize = dot_prod_dict_train_norm is not None

    component_names = dot_prod_dict_train[0].keys()
    total_dot_products_train = np.array([dot_prod_dict_train[example][residual_stream_component_name] for example in range(len(dot_prod_dict_train))])
    total_dot_products_test = np.array([dot_prod_dict_test[example][residual_stream_component_name] for example in range(len(dot_prod_dict_test))])
    if normalize:
        total_norm_train = np.array([dot_prod_dict_train_norm[example][residual_stream_component_name] for example in range(len(dot_prod_dict_train))])
        total_norm_test = np.array([dot_prod_dict_test_norm[example][residual_stream_component_name] for example in range(len(dot_prod_dict_test))])

        # Normalize the total dot products
        total_dot_products_train /= total_norm_train
        total_dot_products_test /= total_norm_test

    r2s = dict()
    for component_name in component_names:
        component_norm_train = np.array([dot_prod_dict_train_norm[example][component_name] for example in range(len(dot_prod_dict_train))])
        component_norm_test = np.array([dot_prod_dict_test_norm[example][component_name] for example in range(len(dot_prod_dict_test))])

        if normalize:
            component_dot_products_train = np.array([dot_prod_dict_train[example][component_name] for example in range(len(dot_prod_dict_train))])
            component_dot_products_test = np.array([dot_prod_dict_test[example][component_name] for example in range(len(dot_prod_dict_test))])

            # Normalize the component dot products
            component_dot_products_train /= component_norm_train
            component_dot_products_test /= component_norm_test

        lr = LinearRegression()
        lr.fit(component_dot_products_train.reshape(-1, 1), total_dot_products_train)
        pred = lr.predict(component_dot_products_test.reshape(-1, 1))

        r2s[component_name] = r2_score(total_dot_products_test, pred)
    return r2s


def compute_multi_component_prediction_r2(dot_prod_dict_train, dot_prod_dict_test, dot_prod_dict_train_norm = None, dot_prod_dict_test_norm = None, residual_stream_component_name = 'blocks.10.hook_resid_pre'):
    assert dot_prod_dict_train_norm is None or dot_prod_dict_train_norm is not None, "If you provide dot_prod_dict_train_norm, you must also provide dot_prod_dict_test_norm"
    normalize = dot_prod_dict_train_norm is not None

    # Expecting that dot_prod_dict is a nested dict of example: {component: dot product} pairs
    component_names = dot_prod_dict_train[0].keys()
    total_dot_products_train = np.array([dot_prod_dict_train[example][residual_stream_component_name] for example in range(len(dot_prod_dict_train))])
    total_dot_products_test = np.array([dot_prod_dict_test[example][residual_stream_component_name] for example in range(len(dot_prod_dict_test))])
    if normalize:
        total_norm_train = np.array([dot_prod_dict_train_norm[example][residual_stream_component_name] for example in range(len(dot_prod_dict_train))])
        total_norm_test = np.array([dot_prod_dict_test_norm[example][residual_stream_component_name] for example in range(len(dot_prod_dict_test))])

        # Normalize the total dot products
        total_dot_products_train /= total_norm_train
        total_dot_products_test /= total_norm_test

    r2s = {}

    for c1, c2 in itertools.combinations(component_names, 2):
        c1_dot_products_train = np.array([dot_prod_dict_train[example][c1] for example in range(len(dot_prod_dict_train))])
        c1_dot_products_test = np.array([dot_prod_dict_test[example][c1] for example in range(len(dot_prod_dict_test))])

        c2_dot_products_train = np.array([dot_prod_dict_train[example][c2] for example in range(len(dot_prod_dict_train))])
        c2_dot_products_test = np.array([dot_prod_dict_test[example][c2] for example in range(len(dot_prod_dict_test))])

        component_dot_products_train = np.stack((c1_dot_products_train, c2_dot_products_train), axis=1)
        component_dot_products_test = np.stack((c1_dot_products_test, c2_dot_products_test), axis=1)

        if normalize:
            c1_norm_train = np.array([dot_prod_dict_train_norm[example][c1] for example in range(len(dot_prod_dict_train))])
            c1_norm_test = np.array([dot_prod_dict_test_norm[example][c1] for example in range(len(dot_prod_dict_test))])

            c2_norm_train = np.array([dot_prod_dict_train_norm[example][c2] for example in range(len(dot_prod_dict_train))])
            c2_norm_test = np.array([dot_prod_dict_test_norm[example][c2] for example in range(len(dot_prod_dict_test))])

            # Normalize the component dot products
            component_dot_products_train /= np.stack((c1_norm_train, c2_norm_train), axis=1)
            component_dot_products_test /= np.stack((c1_norm_test, c2_norm_test), axis=1)

        pipeline = LinearRegression()
        pipeline.fit(component_dot_products_train.reshape(-1, 1), total_dot_products_train)
        pred = pipeline.predict(component_dot_products_test.reshape(-1, 1))

        r2s[f"{c1} x {c2}"] = r2_score(total_dot_products_test, pred)
    return r2s


def compare_component_diff_in_means(model_bundle: ModelBundle, multicomponent=False):
    train_dict = collections.defaultdict(dict)
    test_dict = collections.defaultdict(dict)

    for pos in range(-1, -MIN_LEN - 1, -1):
        print("pos = ", pos)
        subset_len = len(model_bundle.harmful_inst_train) # There is a great imbalance in dataset size

        model = model_bundle.model
        refusal_direction = model_bundle.refusal_direction

        harmless_outputs_train = get_act(model, model_bundle.harmless_inst_train[:subset_len], pos)
        harmful_outputs_train = get_act(model, model_bundle.harmful_inst_train[:subset_len], pos)
        harmless_outputs_test = get_act(model, model_bundle.harmless_inst_test[:subset_len], pos)
        harmful_outputs_test = get_act(model, model_bundle.harmful_inst_test[:subset_len], pos)

        diff_in_means_train = get_mean_dot_prod(harmful_outputs_train)
        diff_in_means_train = dict_subtraction(diff_in_means_train, get_mean_dot_prod(harmless_outputs_train))
        for component_name in diff_in_means_train:
            diff_in_means_train[component_name] = torch.nn.functional.normalize(diff_in_means_train[component_name])
            diff_in_means_train[component_name] = torch.dot(diff_in_means_train[component_name], refusal_direction.type(diff_in_means_train[component_name].dtype)).item()

        diff_in_means_test = get_mean_dot_prod(harmful_outputs_test)
        diff_in_means_test = dict_subtraction(diff_in_means_test, get_mean_dot_prod(harmless_outputs_test))
        for component_name in diff_in_means_test:
            diff_in_means_test[component_name] = torch.nn.functional.normalize(diff_in_means_test[component_name])
            diff_in_means_test[component_name] = torch.dot(diff_in_means_test[component_name], refusal_direction.type(diff_in_means_test[component_name].dtype)).item()

        # Sort the diff in means by train
        diff_in_means_train_list = list(diff_in_means_train.items())
        diff_in_means_train_list.sort(key=lambda x: abs(x[1]), reverse=True)

        for component_name, _ in diff_in_means_train_list:
            diff_train = diff_in_means_train[component_name]
            diff_test = diff_in_means_test[component_name]

            train_dict[pos][component_name] = diff_train
            test_dict[pos][component_name] = diff_test
            print(f"{component_name}: Diff in means - Train: {diff_train:.4f} Test: {diff_test:.4f}")
            print()


    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)
    train_df.to_csv(os.path.join(model_bundle.results_dir, 'train_df.csv'))
    test_df.to_csv(os.path.join(model_bundle.results_dir, 'test_df.csv'))


def compare_component_prediction_r2(model_bundle: ModelBundle, multicomponent=False):
    train_dict = collections.defaultdict(dict)
    test_dict = collections.defaultdict(dict)
    harmless_dict, harmful_dict = collections.defaultdict(dict), collections.defaultdict(dict)

    compute_component_prediction_r2_func = compute_multi_component_prediction_r2 if multicomponent else compute_component_prediction_r2

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
        harmless_r2s = compute_component_prediction_r2_func(harmless_outputs_train, harmless_outputs_test)
        harmful_r2s = compute_component_prediction_r2_func(harmful_outputs_train, harmful_outputs_test)

        # Sort the diff in means by train
        harmful_r2s_list = list(harmful_r2s.items())
        harmful_r2s_list.sort(key=lambda x: x[1], reverse=True)

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

        for component_names, _ in harmful_r2s_list:
            harmless_dict[pos][component_names] = harmless_r2s[component_names]
            harmful_dict[pos][component_names] = harmful_r2s[component_names]
            print(
                f"{component_names}: R2 - Harmless: {harmless_r2s[component_names]:.4f} Harmful: {harmful_r2s[component_names]:.4f}")

        # del harmless_outputs_train
        # del harmless_outputs_test
        # del harmful_outputs_train
        # del harmful_outputs_test
        # gc.collect()

    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)
    train_df.to_csv(os.path.join(model_bundle.results_dir, 'train_df.csv'))
    test_df.to_csv(os.path.join(model_bundle.results_dir, 'test_df.csv'))

    harmful_df = pd.DataFrame(harmful_dict)
    harmless_df = pd.DataFrame(harmless_dict)
    harmful_df.to_csv(os.path.join(model_bundle.results_dir, 'harmful_r2s.csv'))
    harmless_df.to_csv(os.path.join(model_bundle.results_dir, 'harmless_r2s.csv'))
