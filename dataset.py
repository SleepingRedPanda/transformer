from typing import Any
import torch
from torch import nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_trg, lang_src, lang_trg, seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.lang_src = lang_src
        self.lang_trg = lang_trg

        self_sos_token = torch.tensor(
            [tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self_eos_token = torch.tensor(
            [tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self_pad_token = torch.tensor(
            [tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_trg_pair = self.ds[index]
        text_src = src_trg_pair['translation'][self.lang_src]
        text_trg = src_trg_pair['translation'][self.lang_trg]

        enc_input_tokens = self.tokenizer_src.encode(text_src).ids
        dec_input_tokens = self.tokenizer_trg.encode(text_trg).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Input sequence too long:")

        # Add SOS and EOS tokens to the source sequences
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the decoder label (what we expect from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] *
                             dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # [seq_len]
            "decoder_input": decoder_input, # [seq_len]
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() % causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # [seq_len]
            "text_src": text_src,
            "text_trg": text_trg
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0