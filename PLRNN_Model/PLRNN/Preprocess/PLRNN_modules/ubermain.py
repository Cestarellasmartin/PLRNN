from multitasking import *


def ubermain(n_runs):
    """
    Specify the argument choices you want to be tested here in list format:
    e.g. args.append(Argument('dim_z', [5, 6], add_to_name_as='z'))
    will test for dimensions 5 and 6 and save experiments under z5 and z6
    """
    args = []
    args.append(Argument('experiment', ['example_data']))
    args.append(Argument('data_path', ["example_data/training_data_10.npy"]))
    args.append(Argument('inputs_path', ["example_data/training_inputs_10.npy"]))
    args.append(Argument('use_gpu', [1]))
    args.append(Argument('n_epochs', [5000]))
    args.append(Argument('pretrain_epochs', [2]))
    args.append(Argument('use_reg', [1]))
    args.append(Argument('TF_alpha', [0.1]))
    args.append(Argument('dim_hidden', [200], add_to_name_as="H"))
    args.append(Argument('reg_lambda1', [0]))
    args.append(Argument('reg_lambda2', [0]))
    args.append(Argument('reg_lambda3', [0]))
    args.append(Argument('batch_size', [64]))
    args.append(Argument('batches_per_epoch', [50]))
    args.append(Argument('dim_z', [3]))
    args.append(Argument('enc_dim', [3]))
    args.append(Argument('seq_len', [30]))
    args.append(Argument('learning_rate', [1e-3]))
    args.append(Argument('observation_noise_level', [0.1]))
    args.append(Argument('save_step', [100]))
    args.append(Argument('save_img_step', [100]))
    args.append(Argument('run', list(range(1, 1 + n_runs))))
    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # number of runs for each experiment
    n_runs = 1
    # number of runs to run in parallel
    n_cpu = 1
    # number of processes run parallel on a single GPU
    n_proc_per_gpu = 1

    args = ubermain(n_runs)
    run_settings(*create_tasks_from_arguments(args, n_proc_per_gpu, n_cpu))
