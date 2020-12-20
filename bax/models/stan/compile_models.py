"""
Code to compile stan models
"""

import argparse

from gp_fixedsig import get_stanmodel as get_stanmodel_fixedsig
from gp_fixedsig_distmat import get_stanmodel as get_stanmodel_fixedsig_distmat


def main(model_str):
    """Re-compile model specified by model_str."""
    if model_str == 'gp_fixedsig':
        model = get_stanmodel_fixedsig(recompile=True)
    elif model_str == 'gp_fixedsig_distmat':
        model = get_stanmodel_fixedsig_distmat(recompile=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Stan model compiler')
    parser.add_argument('-m', '--model_str', help='Model string', default='gp_fixedsig')

    args = parser.parse_args()
    assert args.model_str in ['gp_fixedsig', 'gp_fixedsig_distmat']
    print('Compiling Stan model: {}'.format(args.model_str))

    main(args.model_str)
