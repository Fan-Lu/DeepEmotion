import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='DeepEmotion')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# glimpse network params
glimpse_arg = add_argument_group('Glimpse Network Params')
glimpse_arg.add_argument('--patch_size', type=int, default=8,
                         help='size of extracted patch at highest res')
glimpse_arg.add_argument('--glimpse_scale', type=int, default=2,
                         help='scale of successive patches')
glimpse_arg.add_argument('--num_patches', type=int, default=1,
                         help='# of downscaled patches per glimpse')
glimpse_arg.add_argument('--loc_hidden', type=int, default=128,
                         help='hidden size of loc fc')
glimpse_arg.add_argument('--glimpse_hidden', type=int, default=128,
                         help='hidden size of glimpse fc')


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--valid_size', type=float, default=0.1,
                      help='Proportion of training set used for validation')
data_arg.add_argument('--batch_size', type=int, default=32,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--show_sample', type=str2bool, default=False,
                      help='Whether to visualize a sample grid of the data')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--momentum', type=float, default=0.5,
                       help='Nesterov momentum value')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=50,
                       help='Number of epochs to wait before stopping train')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed