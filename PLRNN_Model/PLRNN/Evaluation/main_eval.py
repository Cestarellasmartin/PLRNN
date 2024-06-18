from typing import List

from pandas.core.indexes import numeric
from evaluation.klx_gmm import calc_kl_from_data

import os
import numpy as np
import torch as tc
import pandas as pd
from glob import glob

import utils
from evaluation import mse
from evaluation import klx
from bptt import models
from evaluation.pse import power_spectrum_error, power_spectrum_error_per_dim
from evaluation.klx_gmm import calc_kl_from_data

EPOCH = None
SKIP_TRANS = 0
DATA_GENERATED = None
PRINT = True


def get_generated_data(model, data, inputs, indices):
    """
    Use global variable as a way to draw trajectories only once for evaluating several metrics, for speed.
    :param model:
    :return:
    """
    global DATA_GENERATED
    # Problem: if block is only entered once per training,
    # to trajectory is never updated with better models.
    #if DATA_GENERATED is None:
    # generate each trial
    X_gen_ = []
    for i, (x, s) in enumerate(zip(data, inputs)):
        X_gen, Z = model.generate_free_trajectory(x, s, x.shape[0], i)
        X_gen_.append(X_gen)

    DATA_GENERATED = tc.cat(X_gen_)
    return DATA_GENERATED


def printf(x):
    if PRINT:
        print(x)


class Evaluator(object):
    def __init__(self, init_data, inputs):
        model_ids, data, save_path = init_data
        self.model_ids = model_ids
        self.save_path = save_path
        if isinstance(data, list):
            self.data = [tc.tensor(i[SKIP_TRANS:], dtype=tc.float) for i in data]
            self.inputs = [tc.tensor(i[SKIP_TRANS:], dtype=tc.float) for i in inputs]
        else:
            self.data = tc.tensor(data[SKIP_TRANS:], dtype=tc.float)
            self.inputs = tc.tensor(inputs[SKIP_TRANS:], dtype=tc.float)

        self.name = NotImplementedError
        self.dataframe_columns = NotImplementedError

    def metric(self, model):
        return NotImplementedError

    def evaluate_metric(self):
        metric_dict = dict()
        assert self.model_ids is not None
        for model_id in self.model_ids:
            model = self.load_model(model_id)
            metric_dict[model_id] = self.metric(model)
        self.save_dict(metric_dict)

    def load_model(self, model_id):
        model = models.Model()
        model.init_from_model_path(model_id, EPOCH)
        model.eval()
        print(model.get_num_trainable())
        return model

    def save_dict(self, metric_dict):
        df = pd.DataFrame.from_dict(data=metric_dict, orient='index')
        df.columns = self.dataframe_columns
        utils.make_dir(self.save_path)
        df.to_csv('{}/{}.csv'.format(self.save_path, self.name), sep='\t')


class EvaluateKLx(Evaluator):
    def __init__(self, init_data, inputs):
        super(EvaluateKLx, self).__init__(init_data, inputs)
        self.name = 'klx'
        self.dataframe_columns = ('klx',)
        #self.inputs = inputs

    def metric(self, model, indices):
        data = [t.to(model.device) for t in self.data]
        inputs = [t.to(model.device) for t in self.inputs]
        data_gen = get_generated_data(model, data, inputs, indices).cpu()
        data_cat = tc.cat(self.data).cpu()

        _, dx = self.data[0].size()
        if dx > 5:
            #klx_value = klx.klx_metric(*self.pca(data_gen, self.data)).cpu()
            klx_value = calc_kl_from_data(data_gen, data_cat)
        else:
            klx_value = klx.klx_metric(data_gen, data_cat)

        printf('\tKLx {}'.format(klx_value.item()))
        return [np.array(klx_value.numpy())]

class EvaluateMSE(Evaluator):
    def __init__(self, init_data, inputs):
        super(EvaluateMSE, self).__init__(init_data, inputs)
        self.name = 'mse'
        self.n_steps = 25
        #self.indices = indices
        self.dataframe_columns = tuple(['{}'.format(i) for i in range(1, 1 + self.n_steps)])

    def metric(self, model, indices):
        if isinstance(self.data, list):
            mse_results = mse.n_steps_ahead_pred_mse(model, self.data[0], self.inputs[0], n_steps=self.n_steps, indices= indices)
        else:
            mse_results = mse.n_steps_ahead_pred_mse(model, self.data, self.inputs, n_steps=self.n_steps, indices = indices )
        for step in [1, 5, 25]:
            printf('\tMSE-{} {}'.format(step, mse_results[step-1]))
        return mse_results


class EvaluatePSE(Evaluator):
    def __init__(self, init_data, inputs):
        super(EvaluatePSE, self).__init__(init_data, inputs)
        self.name = 'pse'
        if isinstance(self.data, list):
            n_dim = self.data[0].shape[1]
        else:
            n_dim = self.data.shape[1]
        self.dataframe_columns = tuple(['mean'] + ['dim_{}'.format(dim) for dim in range(n_dim)])

    def metric(self, model, indices):
        data = [t.to(model.device) for t in self.data]
        inputs = [t.to(model.device) for t in self.inputs]
        data_gen = get_generated_data(model, data, inputs, indices).cpu()
        data_cat = tc.cat(self.data).cpu()

        x_gen = data_gen.numpy()
        x_true = data_cat.numpy()
        pse, pse_per_dim = power_spectrum_error(x_gen=x_gen, x_true=x_true)

        printf('\tPSE {}'.format(pse))
        printf('\tPSE per dim {}'.format(pse_per_dim))
        return [pse] + pse_per_dim


class SaveArgs(Evaluator):
    def __init__(self, init_data):
        super(SaveArgs, self).__init__(init_data)
        self.name = 'args'
        self.dataframe_columns = ('dim_x', 'dim_z', 'dim_s', 'n_bases')

    def metric(self, model):
        args = model.args
        return [args['dim_x'], args['dim_z'], args['dim_s'], args['n_bases']]


def gather_eval_results(eval_dir='save', save_path='save_eval', metrics=None):
    """Pre-calculated metrics in individual model directories are gathered in one csv file"""
    if metrics is None:
        metrics = ['klx', 'pse']
    metrics.append('args')
    model_ids = get_model_ids(eval_dir)
    for metric in metrics:
        paths = [os.path.join(model_id, '{}.csv'.format(metric)) for model_id in model_ids]
        data_frames = []
        for path in paths:
            try:
                data_frames.append(pd.read_csv(path, sep='\t', index_col=0))
            except:
                print('Warning: Missing model at path: {}'.format(path))
        data_gathered = pd.concat(data_frames)
        utils.make_dir(save_path)
        metric_save_path = '{}/{}.csv'.format(save_path, metric)
        data_gathered.to_csv(metric_save_path, sep='\t')


def choose_evaluator_from_metric(metric_name, init_data, inputs):
    if metric_name == 'mse':
        EvaluateMetric = EvaluateMSE(init_data, inputs)
    elif metric_name == 'klx':
        EvaluateMetric = EvaluateKLx(init_data, inputs)
    elif metric_name == 'pse':
        EvaluateMetric = EvaluatePSE(init_data, inputs)
    else:
        raise NotImplementedError
    return EvaluateMetric


def eval_model_on_data_with_metric(model, data, inputs, metric, indices):
    init_data = (None, data, None)
    #init_inputs = (None, inputs, None)
    EvaluateMetric = choose_evaluator_from_metric(metric, init_data, inputs)
    #EvaluateMetric.data = data
    metric_value = EvaluateMetric.metric(model, indices)
    return metric_value


def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('/')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers


def get_model_ids(path):
    """
    Get model ids from a directory by recursively searching all subdirectories for files ending with a number
    """
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids


def eval_model(args):
    save_path = args.load_model_path
    evaluate_model_path(args, model_path=save_path, metrics=args.metrics)


def evaluate_model_path(data_path, inputs_path, model_path=None, metrics=None):
    """Evaluate a single model in directory model_path w.r.t. metrics and save results in csv file in model_path"""
    model_ids = [model_path]
    data = utils.read_data(data_path)
    inputs = utils.read_data(inputs_path)
    init_data = (model_ids, data, model_path)
    Save = SaveArgs(init_data)
    Save.evaluate_metric()
    global DATA_GENERATED
    DATA_GENERATED = None

    for metric_name in metrics:
        EvaluateMetric = choose_evaluator_from_metric(metric_name=metric_name, init_data=(model_ids, data, model_path), inputs=inputs)
        EvaluateMetric.evaluate_metric()


def evaluate_all_models(eval_dir, data_path, inputs_path, metrics):
    model_paths = get_model_ids(eval_dir)
    n_models = len(model_paths)
    print('Evaluating {} models'.format(n_models))
    for i, model_path in enumerate(model_paths):
        print('{} of {}'.format(i+1, n_models))
        # try:
        evaluate_model_path(data_path=data_path, inputs_path=inputs_path,  model_path=model_path, metrics=metrics)
        # except:
        #     print('Error in model evaluation {}'.format(model_path))
    return

def print_metric_stats(save_path: str, metrics: List):
    # MSE
    path = os.path.join(save_path, 'mse.csv')
    df = pd.read_csv(path, delimiter='\t')
    mse5 = (df.mean(0, numeric_only=True)['5'], df.std(numeric_only=True)['5'])
    mse10 = (df.mean(0, numeric_only=True)['10'], df.std(numeric_only=True)['10'])
    mse20 = (df.mean(0, numeric_only=True)['20'], df.std(numeric_only=True)['20'])

    #PSE
    path = os.path.join(save_path, 'pse.csv')
    df = pd.read_csv(path, delimiter='\t')
    pse = (df.mean(0, numeric_only=True)['mean'], df.std(numeric_only=True)['mean'])

    # Dstsp
    path = os.path.join(save_path, 'klx.csv')
    df = pd.read_csv(path, delimiter='\t')
    df_sub = df['klx']
    df_sub = df_sub[df_sub > 0]
    klx = (df_sub.mean(0), df_sub.std(0))

    new_df = pd.DataFrame({
        '5-MSE': mse5,
        '10-MSE': mse10,
        '20-MSE': mse20,
        'PSC': pse,
        'KLX': klx
    })
    new_df.to_csv(os.path.join(save_path, 'stats1.csv'), sep='\t')



#if __name__ == '__main__':
    #eval_dir = 'results/10_runs_fix_obs_model_no_reg_vienna/z_141_TF_20/001'
    #data_path = 'datasets/df1.npy'
    #metrics = ['mse', 'klx', 'pse']
    #evaluate_all_models(eval_dir=eval_dir, data_path=data_path, inputs_path=inputs_path, metrics=metrics)
    #gather_eval_results(eval_dir=eval_dir, metrics=metrics)

if __name__ == '__main__':
    # tc.set_num_threads(4)
    eval_dirs = [
        "results/LORTDE-PLRNN"

        # "results/LORLD-PLRNN",

    ]
    for eval_dir in eval_dirs:
        # eval_dir = 'results1/BSN-BIG-reg-obs/B47dz26tau05sq500lambda0.1'
        metrics = ['pse', 'mse', 'klx']

        data_path = 'datasets/Lorenz_TDE/lorenz_pn.01_on.01_test_tde3_10_tf.npy'

        evaluate_all_models(eval_dir=eval_dir, data_path=data_path, metrics=metrics)
        gather_eval_results(eval_dir=eval_dir, save_path=eval_dir.split("/")[-1], metrics=metrics)
        # eval_dirs = ['LOR-TDE', 'BSN-BIG', 'LOR-BIG', 'LOR96-upsample', 'LOR96', 'NP', 'LOR-noisy', 'LOR-lowdata']
        # for eval_dir in eval_dirs:
        print(print_metric_stats(eval_dir.split("/")[-1], metrics))
