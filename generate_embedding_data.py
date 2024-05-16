from data.embedding_generator import EmbeddingGenerator

from argparse import ArgumentParser
import multiprocessing as mp

data_files = {
    'netflix_prize' : {
        'train' : [f'train_{i}' for i in range(8)],
        'test' : [f'test_{i}' for i in range(2)]
    },
    'movielens_25m' : {
        'train' : [f'train_{i}' for i in range(4)],
        'test' : [f'test_{i}' for i in range(1)]
    },
    'yahoo_r2' : {
        'train' : [f'train_{i}' for i in range(10)],
        'test' : [f'test_{i}' for i in range(10)]
    }
}

def generate_data(data_name: str, data_files: str, used_models: list[str], context_size: int):
    print(f"Generating data for {data_name} - {data_files}")
    generator = EmbeddingGenerator(data_name, data_files)
    if 'skip_gram' in used_models:
        generator.generate_data('skip_gram', generator.users, context_size, True)
    if 'cbow' in used_models:
        generator.generate_data('cbow', generator.users, context_size, True)



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, nargs='+', required=False, choices=data_files.keys(),
        default=data_files.keys(), help='Dataset name'
    )
    parser.add_argument(
        '--filename', type=str, required=False, default='all',
        help="File name. 'train' : all train files, 'test' : all test files, 'all' : all files"
    )
    parser.add_argument(
        '--used_model', type=str, nargs='+', choices=['skip_gram', 'cbow'], required=False,
        default=['skip_gram', 'cbow'],
        help='Model to be used for embedding generation'
    )
    parser.add_argument(
        '--context_size', type=int, required=False, default=2,
        help='Context size for embedding generation'
    )
    parser.add_argument(
        '--use_multiprocessing', type=bool, required=False, default=False
    )

    args = vars(parser.parse_args())


    for dataset_name in args['dataset']:

        files: list[str]
        if args['filename'] == 'all':
            files = data_files[dataset_name]['train'] + data_files[dataset_name]['test']
        elif args['filename'] == 'train':
            files = data_files[dataset_name]['train']
        elif args['filename'] == 'test':
            files = data_files[dataset_name]['test']
        else:
            files = [args['filename']]

        if args['use_multiprocessing']:
            # processes = [
            #     mp.Process(target=generate_data, args=(dataset_name, file_name, args['used_model'], args['context_size']))
            #     for file_name in data_files[dataset_name]['train']
            # ]

            # for p in processes:
            #     p.start()
            # for p in processes:
            #     p.join()
            pool = mp.Pool(mp.cpu_count())
            var = [(dataset_name, file_name, args['used_model'], args['context_size']) for file_name in files]
            pool.starmap(generate_data, var)
            pool.close()
            pool.join()
        else:
            for file in files:
                generate_data(dataset_name, file, args['used_model'], args['context_size'])
                print(f"Generated data for {dataset_name} - {file}")
        print(f"Generated data for {dataset_name}")