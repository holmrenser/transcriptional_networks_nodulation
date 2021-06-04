#!/usr/bin/env python
import pandas as pd
from arboreto.algo import diy
from distributed import Client

import argparse

REGRESSOR_TYPE = 'GBM'
SEED = 42

"""
The following are parameter values to test for optimizing
The first value to be tested is always the default
All combinations will be tested
"""
LOSS = ['ls', 'lad']
LEARNING_RATE = [0.01, 0.001]
MAX_FEATURES = [0.1, 0.5, 'auto']
SUBSAMPLE = [0.9, 0.7, 0.99]
EARLYSTOP_WINDOW_LENGTH = [25, 100]

def run_inference(
    expression_data, tf_names, regressor_kwargs,
    out_file, earlystop_window_length, n_workers, threads_per_worker
    ):
    print('starting scheduler')
    client = Client(
        n_workers = n_workers,
        threads_per_worker = threads_per_worker,
        memory_limit='48GB'
    )

    try:
        network = diy(
            expression_data = expression_data, tf_names = tf_names,
            client_or_address = client, verbose = True, seed = SEED,
            early_stop_window_length = earlystop_window_length,
            regressor_type = REGRESSOR_TYPE, regressor_kwargs = regressor_kwargs
        )

        network.to_csv(
            out_file, sep = '\t', header = False, index = False
        )
    except Exception as e:
        print('Module inference error')
        print(e)
    finally:
        client.close()


def main(
    transcriptome_file, regulator_file, species, out_file_prefix, 
    n_workers, threads_per_worker
    ):
    print('reading data')

    tf_info = pd.read_csv(regulator_file, sep = '\t', index_col = 0)
    tf_names = list(tf_info.loc[tf_info['species'] == species].index)

    df = pd.read_csv(transcriptome_file, sep = '\t', index_col = 0)
    
    num_not_expressed = (df.std(axis = 1) == 0).sum()

    print(f'removing {num_not_expressed} genes that have zero \
        expression in all samples')
    #Filter genes that are not expressed
    df = df.loc[df.std(axis = 1) > 0]

    expression_data = df.T

    for loss in LOSS:
        for learning_rate in LEARNING_RATE:
            for max_features in MAX_FEATURES:
                for subsample in SUBSAMPLE:
                    for earlystop_window_length in EARLYSTOP_WINDOW_LENGTH:
                        regressor_kwargs = dict(
                            loss = loss,
                            learning_rate = learning_rate,
                            max_features = max_features,
                            subsample = subsample
                        )
                        out_file = (
                            f'{out_file_prefix}_'
                            f'[loss={loss}]_'
                            f'[learning_rate={learning_rate}]_'
                            f'[max_features={max_features}]_'
                            f'[subsample={subsample}]_'
                            f'[earlystop={earlystop_window_length}]'
                            '.network.tsv'
                        )
                        print(f'out_file: {out_file}')
                        run_inference(
                            expression_data = expression_data,
                            regressor_kwargs = regressor_kwargs,
                            tf_names = tf_names,
                            out_file = out_file,
                            earlystop_window_length = earlystop_window_length, n_workers = n_workers,
                            threads_per_worker = threads_per_worker
                        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gene Regulatory \
        Network inference from transcriptome sequencing')
    parser.add_argument('-t', '--transcriptome-file', required = True)
    parser.add_argument('-r', '--regulator-file', required = True)
    parser.add_argument('-o','--out-file-prefix', required = True)
    parser.add_argument('-s','--species', required = True)
    parser.add_argument('--n-workers', type = int, default = 1)
    parser.add_argument('--threads-per-worker', type = int, default = 1)

    args = parser.parse_args()

    main(**vars(args))