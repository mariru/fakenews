import argparse

def parse_args():
        parser = argparse.ArgumentParser(description="run exponential family embeddings on text")

        parser.add_argument('--K', type=int, default=20,
                            help='Number of dimensions. Default is 100.')

        parser.add_argument('--sig', type=float, default = 100.0,
                            help='Regularization on global embeddings. Default is 100.')

        parser.add_argument('--n_iter', type=int, default = 10,
                            help='Number of passes over the data. Default is 10.')

        parser.add_argument('--n_epochs', type=int, default=1000,
                            help='Number of epochs. Default is 100.')

        parser.add_argument('--cs', type=int, default=4,
                            help='Context size. Default is 4.')

        parser.add_argument('--ns', type=int, default=10,
                            help='Number of negative samples. Default is 10.')

        parser.add_argument('--completely_separate', type=bool, default=False,
                            help='Train each slice separately. Default is False.')

        parser.add_argument('--separate', type=bool, default=False,
                            help='Train each slice separately. Default is False.')


        parser.add_argument('--debug', type=bool, default=False,
                            help='Debug mode (10th of data). Default is False.')

        parser.add_argument('--init', type=str, default='',
                            help='Folder name to load variational.dat for initialization. Default is \'\' for no initialization')

        parser.add_argument('--fpath', type=str, default='../dat/fake/',
                            help='path to data (arxiv small)')

        args =  parser.parse_args()
        print(args)
        return args
