from datasets import load_metric
import language_tool_python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sys
sys.path.append('/data/private/GPT/gpt_lm')
from korgpt2 import KorGPT2Tokenizer

from KoBERTScore import BERTScore

import torch
import torch.nn.functional as F

class MetFunc():
    def __init__(self):
        model_path = "/data/private/GPT/openai-gpt2/medium"
        self.gpt_model = GPT2LMHeadModel.from_pretrained(model_path).cuda() # gpt2-medium
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_path) # gpt2-medium
        self.gpt_model.eval()
        
        self.bertscore_metric = load_metric('bertscore')
        self.sacrebleu_metric = load_metric('sacrebleu')
        self.wer_metric = load_metric("wer")
        self.bleurt_metric = load_metric("bleurt") # bleurt-large-51
        self.grammar_metric = language_tool_python.LanguageTool('en-US')
        
    def CalBertScore(self, src_text, generations):
        references = [src_text for _ in range(len(generations))]
        score = self.bertscore_metric.compute(predictions=generations, references=references, lang="en")
        
        bertscore_list = score['f1']
        return bertscore_list
    
    def CalSacreBleu(self, src_text, generations):
        sacrebleu_list = []
        for tgt_text in generations:
            sacrebleu_list.append(self.sacrebleu_metric.compute(predictions=[tgt_text], references=[[src_text]])['score'])
            
        return sacrebleu_list
    
    def CalWer(self, src_text, generations):
        wer_list = []
        for tgt_text in generations:
            wer_list.append(self.wer_metric.compute(predictions=[tgt_text], references=[src_text]))
            
        return wer_list
    
    def CalBleurt(self, src_text, generations):
        references = [src_text for _ in range(len(generations))]
        bleurt_list = self.bleurt_metric.compute(predictions=generations, references=references)
            
        return bleurt_list['scores']
    
    def CalGrammar(self, generations):
        grammar_list = []
        for tgt_text in generations:
            matches = self.grammar_metric.check(tgt_text)
            grammar_list.append(len(matches))

        return grammar_list
    
    def CalPPL(self, generations):
        PPL_list = []
        for generation in generations:
            lm_tokens = self.gpt_tokenizer.encode(generation, return_tensors="pt").cuda()
            output = self.gpt_model(lm_tokens)
            logit = output[0].squeeze(0)
            labels = lm_tokens.squeeze(0)[1:]
            preds = logit[:-1,:]
            
            loss = F.cross_entropy(preds, labels)
            # calculating perplexity
            perplexity = torch.exp(loss)
            PPL_list.append(perplexity.item())
            
        return PPL_list
    
class MetFuncKo():
    def __init__(self):
        gpt_path = "/data/private/GPT/gpt_lm/model/kor-gpt2/medium"
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_path).cuda() # gpt2-medium
        
        KorGPT2Tokenizer.max_len = 300
        KorGPT2Tokenizer.max_len_single_sentence = 300
        KorGPT2Tokenizer.max_len_sentences_pair = 300
        self.gpt_tokenizer = KorGPT2Tokenizer.from_pretrained(gpt_path) # gpt2-medium
        self.gpt_model.eval()
        
        bert_name = "beomi/kcbert-base"        
        self.KorBertscore = BERTScore(bert_name, best_layer=4)        
        
        self.sacrebleu_metric = load_metric('sacrebleu')
        self.wer_metric = load_metric("wer")
        
    def CalBertScore(self, src_text, generations):
        references = [src_text for _ in range(len(generations))]
        bertscore_list = self.KorBertscore(references=references, candidates=generations, batch_size=128)
        return bertscore_list
    
    def CalSacreBleu(self, src_text, generations):
        sacrebleu_list = []
        for tgt_text in generations:
            sacrebleu_list.append(self.sacrebleu_metric.compute(predictions=[tgt_text], references=[[src_text]])['score'])
            
        return sacrebleu_list
    
    def CalWer(self, src_text, generations):
        wer_list = []
        for tgt_text in generations:
            wer_list.append(self.wer_metric.compute(predictions=[tgt_text], references=[src_text]))
            
        return wer_list    
    
    def CalPPL(self, generations):
        PPL_list = []
        for generation in generations:
            lm_tokens = self.gpt_tokenizer.encode(generation, return_tensors="pt").cuda()
            output = self.gpt_model(lm_tokens)
            logit = output[0].squeeze(0)
            labels = lm_tokens.squeeze(0)[1:]
            preds = logit[:-1,:]
            
            loss = F.cross_entropy(preds, labels)
            # calculating perplexity
            perplexity = torch.exp(loss)
            PPL_list.append(perplexity.item())
            
        return PPL_list    
    