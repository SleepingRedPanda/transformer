import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang] 

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config['lang_src']}-{config['lang_trg']}', split='train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])    
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config['lang_trg'])
    
    # Keep 90% for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_train = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_train])
