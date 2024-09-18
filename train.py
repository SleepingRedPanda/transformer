import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_config, get_weigths_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_trg, max_len, device):
    sos_idx = tokenizer_trg.token_to_id("[SOS]")
    eos_idx = tokenizer_trg.token_to_id("[EOS]")
    
    # Precompute encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True: 
        if decoder_input.size(1) >= max_len:
            break
        
        # Compute the decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Compute the decoder output
        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)

        # Get the next token
        prob = model.project(out[:,-1])
        # Select the token with the highest probability
        _, next_word = torch.max(prob, dim = 1)
        decoder_input = torch.cat([decoder_input, torch.empty(1.1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0)
    
def run_validation(model, validation_ds, tokenizer_src, tokenizer_trg, max_len, device, print_msg, global_state, writer, num_exampes=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    console_width = 80
    
    with torch.no_grad():   
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_trg, max_len, device)
            
            source_text = batch["src_text"][0]
            target_text = batch["trg_text"][0]
            model_output_text = tokenizer_trg.decode(model_output.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)
            
            # Print the results to the console
            
            print_msg(f"-"*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_output_text}")
            
            if count == num_exampes:
                break
        
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang] 

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
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
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_trg']}", split='train')
    
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])    
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config['lang_trg'])
    
    # Keep 90% for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_train = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_train])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'], config['seq_len'])
    
    max_len_src = 0
    max_len_trg = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trg_ids = tokenizer_trg.encode(item['translation'][config['lang_trg']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))
            
    print(f"Max source sequence length: {max_len_src}")
    print(f"Max target sequence length: {max_len_trg}")
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg

def get_model(config, vocab_size, vocab_trg_size):
    model = build_transformer(vocab_size, vocab_trg_size, config["seq_len"], config["seq_len"], config["d_model"])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weigths_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)
            
            label = batch['label'].to(device) # (batch_size, seq_len)
            
            # (batch_size, seq_len, vocab_size) --> (batch_size * seq_len, vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device, lambda msg:batch_iterator.write(msg), global_step, writer)
            
        # save model at the end of each epoch
        model_filename = get_weigths_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)