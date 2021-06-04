#!/usr/bin/env python
import pandas as pd
from arboreto.algo import grnboost2
from distributed import LocalCluster, Client

import argparse

def main(transcriptome_file, regulator_file, species, out_file, 
    n_workers, threads_per_worker):
    print('reading data')

    tf_info = pd.read_csv(regulator_file, sep = '\t', index_col = 0)
    tf_names = list(tf_info.loc[tf_info['Species'] == species].index)

    df = pd.read_csv(transcriptome_file, sep = '\t', index_col = 0)
    
    num_not_expressed = (df.std(axis = 1) == 0).sum()

    print(f'removing {num_not_expressed} genes that have zero expression in all samples')
    #Filter genes that are not expressed
    df = df.loc[df.std(axis = 1) > 0]

    print('starting scheduler')
    client = Client(n_workers = n_workers,
                    threads_per_worker = threads_per_worker,
                    memory_limit='48GB')

    try:
        network = grnboost2(expression_data = df.T, tf_names = tf_names,
                        client_or_address = client, verbose = True)

        network.to_csv(out_file, sep = '\t', header = False,
                    index = False)
    except Exception as e:
        print('Module inference error')
        print(e)
    finally:
        client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gene Regulatory Network inference from transcriptome sequencing')
    parser.add_argument('-t', '--transcriptome-file', required = True)
    parser.add_argument('-r', '--regulator-file', required = True)
    parser.add_argument('-o','--out-file', required = True)
    parser.add_argument('-s','--species', required = True)
    parser.add_argument('--n-workers', type = int, default = 1)
    parser.add_argument('--threads-per-worker', type = int, default = 1)

    args = parser.parse_args()

    main(**vars(args))
