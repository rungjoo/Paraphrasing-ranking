# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch

import pdb
import argparse, logging

from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer

    
class ParaFunc():
    def __init__(self, num_beams=10, no_repeat_ngram_size=3, num_return_sequences=5):
        super(ParaFunc, self).__init__()
#         self.m2m_model_base = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M').cuda() # M2M_large
#         self.m2m_tokenizer_base = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M') # # M2M_large

#         self.m2m_model_large = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M_large').cuda() # M2M_large
#         self.m2m_tokenizer_large = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M_large') # # M2M_large
        
        """Beam search"""
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_return_sequences = num_return_sequences
    
    def same_enc_dec(self, model, tokenizer, src_text, src_lang, variation=False, one_generation=False):
        tokenizer.src_lang = src_lang
        src_tokens = tokenizer.encode(src_text, return_tensors="pt").cuda()

        if not variation:
            encoder_no_repeat = max(2, int(src_tokens.shape[1]/2))
        else:
            encoder_no_repeat = 2
        tgt_token = model.generate(
            src_tokens, 
            max_length=src_tokens.shape[1]+10, 
            num_beams=self.num_beams,
            early_stopping=True,
            no_repeat_ngram_size = self.no_repeat_ngram_size,
            num_return_sequences = self.num_return_sequences,
            encoder_no_repeat_ngram_size = encoder_no_repeat,
        #     num_beam_groups = 2,
        #     diversity_penalty = 0.2,
            forced_bos_token_id=tokenizer.get_lang_id(src_lang)
        )
        tgt_text = tokenizer.batch_decode(tgt_token, skip_special_tokens=True)
        
        if one_generation:
            return tgt_text ## one_generation
    
        src_chars = ''.join(src_text.split(' ')).lower()
        cands = []
        for tgt_text_sample in tgt_text:
            tgt_chars = ''.join(tgt_text_sample.split(' ')).lower()
            if (src_chars != tgt_chars) and (tgt_text_sample not in cands):
                cands.append(tgt_text_sample)
        return cands
    
    def pivoting(self, model, tokenizer, src_text, src_lang, tgt_lang, variation=False):
        tokenizer.src_lang = src_lang
        src_tokens = tokenizer.encode(src_text, return_tensors="pt").cuda()

        if not variation:
            encoder_no_repeat = max(2, int(src_tokens.shape[1]/2))
        else:
            encoder_no_repeat = 2   
        gen_token = model.generate(
            src_tokens, 
            max_length=src_tokens.shape[1]+10, 
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size = 3,
            num_return_sequences = 1,
            encoder_no_repeat_ngram_size = encoder_no_repeat,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
        )  
        tgt_text = tokenizer.batch_decode(gen_token, skip_special_tokens=True)[0]

        tokenizer.src_lang = tgt_lang
        tgt_tokens = tokenizer.encode(tgt_text, return_tensors="pt").cuda()
        if not variation:
            encoder_no_repeat = max(2, int(tgt_tokens.shape[1]/2))
        else:
            encoder_no_repeat = 2               
        gen_token = model.generate(
            tgt_tokens, 
            max_length=tgt_tokens.shape[1]+10,
            num_beams=3,
            early_stopping=True,
            no_repeat_ngram_size = 3,
            num_return_sequences = 1,
            encoder_no_repeat_ngram_size = encoder_no_repeat,
            output_scores = True,
            forced_bos_token_id=tokenizer.get_lang_id(src_lang)
        )
        pivot_text = tokenizer.batch_decode(gen_token, skip_special_tokens=True)
        return pivot_text[0]    
