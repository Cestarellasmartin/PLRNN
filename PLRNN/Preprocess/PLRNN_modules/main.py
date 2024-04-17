import argparse
import torch as tc
import utils

from bptt import bptt_algorithm

from bptt.PLRNN_model import PLRNN

tc.set_num_threads(4)

def get_parser():
    parser = argparse.ArgumentParser(description="TF RNN Training")
    parser.add_argument('--experiment', type=str, default='regtest')
    parser.add_argument('--name', type=str, default='Lorenz')
    parser.add_argument('--run', type=int, default=None)

    # gpu
    parser.add_argument('--use_gpu', type=int, default=1)
    # cuda:0, cuda:1 etc.
    parser.add_argument('--device_id', type=int, default=0)

    # general settings
    parser.add_argument('--no_printing', type=bool, default=True)
    parser.add_argument('--use_tb', type=bool, default=True)
    parser.add_argument('--metrics', type=list, default=['mse', 'pse'])#, 'klx', 'pse'])
    parser.add_argument('--test', type=bool, default=False) ## set to True if you want to use last trial for testing

    # dataset
    parser.add_argument('--data_path', type=str, default='datasets/training_data_10.npy')
    # TODO: inputs not implemented yet
    parser.add_argument('--inputs_path', type=str, default='datasets/training_inputs_10.npy')
    # resume from a model checkpoint
    parser.add_argument('--load_model_path', type=str, default=None)
    # epoch is inferred if None
    parser.add_argument('--resume_epoch', type=int, default=None)

    # model
    parser.add_argument('--dim_z', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=20)
    parser.add_argument('--W_trial', type=bool, default=True)
    parser.add_argument('--clip_range', '-clip', type=float, default=10)
    parser.add_argument('--model', '-m', type=str, default='PLRNN')
    # specifiy which latent model to choose (only affects model PLRNN)
    parser.add_argument('--latent_model', '-ml', type=str,
                        choices=PLRNN.LATENT_MODELS, default='PLRNN')
    parser.add_argument('--n_bases', '-nb', type=int, default=5)
    #parser.add_argument('--latent_model', '-ml', type=str,
                        #choices=PLRNN.LATENT_MODELS, default='dendr-PLRNN')
    parser.add_argument('--mean_center', '-mc', type=int, default=0)
    parser.add_argument('--learn_z0', '-z0', type=int, default=0)

    # SAE
    parser.add_argument('--enc_dim', type=int, default=3)
    parser.add_argument('--AE_learning_rate', '-alr', type=float, default=1e-3)
    parser.add_argument('--AE_weight_decay', '-AEwd', type=float, default=1e-5)
    parser.add_argument('--pretrain_epochs', '-pe', type=int, default=1)
    parser.add_argument('--observation_noise_level', '-gnl', type=float, default=0.05)
    parser.add_argument('--reconstruction_loss_weight', '-rlw', type=float, default=1.0)
    parser.add_argument('--prediction_loss_weight', '-plw', type=float, default=1e-3)
    parser.add_argument('--latent_loss_weight', '-llw', type=float, default=1e-3)
    parser.add_argument('--autoencoder_type', '-aet', type=str, choices=["Identity", "Linear", "MLP"], default="MLP")
    # BPTT
    parser.add_argument('--TF_alpha', '-tfa', type=float, default=0.1)
    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=50)
    parser.add_argument('--seq_len', '-sl', type=int, default=100)
    parser.add_argument('--save_step', '-ss', type=int, default=100)
    parser.add_argument('--save_img_step', '-si', type=int, default=100)

    # optimization
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', '-n', type=int, default=15000)
    parser.add_argument('--gradient_clipping', '-gc', type=float, default=10.)

    # regularization
    # 0: False, 1: True
    parser.add_argument('--use_reg', '-r', type=int, default=1)
    parser.add_argument('--reg_ratios', '-rr', nargs='*', type=float, default=[1.0])
    parser.add_argument('--reg_ratios_W', '-rW', nargs='*', type=float, default=[1.0])
    parser.add_argument('--reg_lambda1', '-ra', nargs='*', type=float, default=[0])
    parser.add_argument('--reg_lambda2', '-rl', nargs='*', type=float, default=[0])
    parser.add_argument('--reg_lambda3', '-rm', nargs='*', type=float, default=[0])
    parser.add_argument('--reg_norm', '-rn', type=str, choices=['l2', 'l1'], default='l2')
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def train(args):
    # prepare training device
    device = utils.prepare_device(args)

    writer, save_path = utils.init_writer(args)

    args, data_set = utils.load_dataset(args)
    #if args.test:
    args, test_set = utils.load_testset(args)
    ###### need to check
    #args, inputs_set = utils.load_inputs(args)
    #######
    utils.check_args(args)
    #utils.check_args(args)

    utils.save_args(args, save_path, writer)
   # utils.save_args(args, save_path, writer)

    training_algorithm = bptt_algorithm.BPTT(args, data_set, test_set,  writer, save_path, device)
    training_algorithm.train()
    return save_path


def main(args):
    train(args)


if __name__ == '__main__':
    main(get_args())

