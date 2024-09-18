from pathlib import Path


def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 20,
        'lr': 1e-4,
        'lang_src': 'en',
        'lang_trg': 'ru',
        'seq_len': 350,
        'datasource': 'opus_books',
        'd_model': 512,
        'batch_size': 32,
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel'
    }


def get_weigths_path(config, epoch):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
