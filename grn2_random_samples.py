#!/usr/bin/env python
import pandas as pd
from arboreto.algo import grnboost2
from distributed import LocalCluster, Client

import argparse

def main(transcriptome_file, regulator_file, species, out_file_prefix, 
    n_random_samples, n_runs, n_workers, threads_per_worker):
    print('reading data')

    tf_info = pd.read_csv(regulator_file, sep = '\t', index_col = 0)
    tf_names = list(tf_info.loc[tf_info['Species'] == species].index)

    df = pd.read_csv(transcriptome_file, sep = '\t', index_col = 0)

    print('starting scheduler')
    client = Client(n_workers = n_workers,
                    threads_per_worker = threads_per_worker,
                    memory_limit='128GB')
    
    for i in range(n_runs):
        out_file = f'{out_file_prefix}_{i}.tsv'
        subsampled_df = df.sample(n_random_samples, axis = 1, random_state = i)
        
        #Filter genes that are not expressed
        num_not_expressed = (subsampled_df.std(axis = 1) == 0).sum()
        print(
            f'removing {num_not_expressed} genes that have zero', 
            'expression in all samples'
        )
        subsampled_df = subsampled_df.loc[df.std(axis = 1) > 0]

        try:
            network = grnboost2(expression_data = subsampled_df.T,
                tf_names = tf_names, client_or_address = client, verbose = True)

            network.to_csv(out_file, sep = '\t', header = False,
                        index = False)
        except Exception as e:
            print('Module inference error')
            print(e)
    client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gene Regulatory Network inference from transcriptome sequencing')
    parser.add_argument('-t', '--transcriptome-file', required = True)
    parser.add_argument('-r', '--regulator-file', required = True)
    parser.add_argument('-o','--out-file-prefix', required = True)
    parser.add_argument('-s','--species', required = True)
    parser.add_argument('--n-random-samples', type = int, default = 10)
    parser.add_argument('--n-runs', type = int, default = 10)
    parser.add_argument('--n-workers', type = int, default = 1)
    parser.add_argument('--threads-per-worker', type = int, default = 1)

    args = parser.parse_args()

    main(**vars(args))
