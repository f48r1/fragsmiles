import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--num_layers", type=int, default=2,
                           help="Number of LSTM layers")
    model_arg.add_argument("--hidden", type=int, default=512,
                           help="Hidden size")
    model_arg.add_argument("--embedding_size", type=int, default=300,
                           help="Embedding size")
    model_arg.add_argument("--dropout", type=float, default=0.2,
                           help="dropout between LSTM layers except for last")

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=70,
                           help='Number of epochs for model training')
    train_arg.add_argument('--n_batch', type=int, default=512,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=3,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    #Sampling
    sample_arg = parser.add_argument_group('Sampling')
    sample_arg.add_argument("--temp",
                            type=float, default=1.0,
                            help="Sampling temperature for softmax")
    sample_arg.add_argument('--onlyNovels',
                           default=False, action='store_true',
                           help='If to generate only novel molecules')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]