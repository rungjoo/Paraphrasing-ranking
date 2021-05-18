import pdb
from datasets import load_metric
import torch
import torch.nn.functional as F
import sys
sys.argv = sys.argv[:1]

from transformers import GPT2LMHeadModel, GPT2Tokenizer   
model_path = "/data/private/GPT/openai-gpt2/medium"
gpt_model = GPT2LMHeadModel.from_pretrained(model_path).cuda() # gpt2-medium
gpt_model.eval()
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_path) # gpt2-medium

"""점수 평가"""
bertscore_metric = load_metric('bertscore')
bleurt_metric = load_metric("bleurt") # bleurt-large-51
sacrebleu_metric = load_metric('sacrebleu')
wer_metric = load_metric("wer")
def score(predictions, references, input_sentences):
    predictions = [x.lower() for x in predictions]
    
    with torch.no_grad():
        """fluency evaluation"""
        PPL_list = []
        for generation in predictions:
            lm_tokens = gpt_tokenizer.encode(generation, return_tensors="pt").cuda()
            output = gpt_model(lm_tokens)
            logit = output[0].squeeze(0)
            labels = lm_tokens.squeeze(0)[1:]
            preds = logit[:-1,:]

            loss = F.cross_entropy(preds, labels)
            # calculating perplexity
            perplexity = torch.exp(loss)
            PPL_list.append(perplexity.item())
        ppl_score = sum(PPL_list)/len(PPL_list)    

        """semantic evaluation"""
        bertscore_list = bertscore_metric.compute(predictions=predictions, references=references, lang="en")['f1']
        bleurt_list = bleurt_metric.compute(predictions=predictions, references=references)['scores']    
        bertscore = sum(bertscore_list)/len(bertscore_list)
        bleurt = sum(bleurt_list)/len(bleurt_list)

        """diversity evaluation"""
        isacrebleu_list = []
        wer_list = []
        for pred, ref in zip(predictions, input_sentences):
            isacrebleu_list.append(100-sacrebleu_metric.compute(predictions=[pred], references=[[ref]])['score'])
            wer_list.append(wer_metric.compute(predictions=[pred], references=[ref]))
        isacrebleu = sum(isacrebleu_list)/len(isacrebleu_list)
        wer = sum(wer_list)/len(wer_list)
    
    return bertscore, bleurt, isacrebleu, wer, ppl_score

def count_overlap(predictions, input_sentences):    
    """overlap evaluation"""
    count = 0
    for pred, inp in zip(predictions, input_sentences):
        pred = pred.lower()
        inp = inp.lower()
        pred_chars = ''.join(pred.split(' ')).lower()
        inp_chars = ''.join(inp.split(' ')).lower()
        if pred_chars == inp_chars:
            count += 1
    return count

def main():
    dataset = 'QQP'
    print("dataset: ", dataset)
        
    """테스트세트 받기"""
    if dataset == 'QQP':
        f = open('../data/QQP_test.txt')
        testset = f.readlines()
        f.close()
        input_sentences = []
        gold_sentences = []
        for line in testset:
            ori, gold = line.split('\t')
            ori, gold = ori.strip(), gold.strip()
            input_sentences.append(ori)
            gold_sentences.append(gold)
    elif dataset == "medical":
        f = open('../data/medical.txt')
        testline = f.readlines()
        f.close()
        input_sentences = []
        gold_sentences = []
        for line in testline:
            ori = line.split('\t')[0].strip()
            gold = line.split('\t')[1].strip()
            input_sentences.append(ori)
            gold_sentences.append(gold)

    """결과 받기"""  
    if dataset == 'QQP':
        f = open('../data/results/PQG/PQG_edlp.txt')
        edlp = f.readlines()
        f.close()
        edlp = [x.strip() for x in edlp]    

        f = open('../data/results/PQG/PQG_edlps.txt')
        edlps = f.readlines()
        f.close()
        edlps = [x.strip() for x in edlps]

        f = open('../data/results/UPSA/UPSA.txt')
        UPSA = f.readlines()
        f.close()
        UPSA = [x.strip() for x in UPSA]

        f = open('../data/results/CGMH/output10.txt')
        CGMH_10 = f.readlines()
        f.close()
        CGMH_10 = [x.strip() for x in CGMH_10]    

        f = open('../data/results/CGMH/output50.txt')
        CGMH_50 = f.readlines()
        f.close()
        CGMH_50 = [x.strip() for x in CGMH_50]

        f = open('../data/results/extra/ours.txt')
        ours = f.readlines()
        f.close()
        ours = [x.strip() for x in ours]

        f = open('../data/results/extra/ours_v2.txt')
        ours_v2 = f.readlines()
        f.close()
        ours_v2 = [x.strip() for x in ours_v2]

        f = open('../data/results/ours_v3.txt')
        ours_v3 = f.readlines()
        f.close()
        ours_v3 = [x.strip() for x in ours_v3]

        f = open('../data/results/M2M/M2M.txt')
        M2M = f.readlines()
        f.close()
        M2M = [x.strip() for x in ours_one]

        gold_sentences = [x.lower() for x in gold_sentences]
        input_sentences = [x.lower() for x in input_sentences] 
    elif dataset == "medical":
        f = open('../data/results/UPSA/UPSA_medical.txt')
        UPSA = f.readlines()
        f.close()
        UPSA = [x.strip() for x in UPSA]

        f = open('../data/results/CGMH/CGMH_medical10.txt')
        CGMH_10 = f.readlines()
        f.close()
        CGMH_10 = [x.strip() for x in CGMH_10]    

        f = open('../data/results/CGMH/CGMH_medical50.txt')
        CGMH_50 = f.readlines()
        f.close()
        CGMH_50 = [x.strip() for x in CGMH_50]        
        
        f = open('../data/results/ours_medical.txt')
        ours_medical = f.readlines()
        f.close()
        ours_medical = [x.strip() for x in ours_medical]   
        
        gold_sentences = [x.lower() for x in gold_sentences]
        input_sentences = [x.lower() for x in input_sentences]         
    
    """계산"""    
    fr = open('./scores.txt','a')
    if dataset == 'QQP':
        fr.write("#################### QQP ####################"+'\n')
        edlp_bertscore, edlp_bleurt, edlp_isacrebleu, edlp_wer, edlp_ppl = score(edlp, gold_sentences, input_sentences)
        edlps_bertscore, edlps_bleurt, edlps_siacrebleu, edlps_wer, edlps_ppl = score(edlps, gold_sentences, input_sentences)
        fr.write("edlp ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(edlp_bertscore, edlp_bleurt, edlp_isacrebleu, edlp_wer, edlp_ppl)+'\n')
        fr.write("edlps ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(edlps_bertscore, edlps_bleurt, edlps_siacrebleu, edlps_wer, edlps_ppl)+'\n')

        UPSA_bertscore, UPSA_bleurt, UPSA_isacrebleu, UPSA_wer, UPSA_ppl = score(UPSA, gold_sentences, input_sentences)
        CGMH_10_bertscore, CGMH_10_bleurt, CGMH_10_isacrebleu, CGMH_10_wer, CGMH_10_ppl = score(CGMH_10, gold_sentences, input_sentences)
        CGMH_50_bertscore, CGMH_50_bleurt, CGMH_50_isacrebleu, CGMH_50_wer, CGMH_50_ppl = score(CGMH_50, gold_sentences, input_sentences)
        fr.write("UPSA ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(UPSA_bertscore, UPSA_bleurt, UPSA_isacrebleu, UPSA_wer, UPSA_ppl)+'\n')
        fr.write("CGMH_10 ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(CGMH_10_bertscore, CGMH_10_bleurt, CGMH_10_isacrebleu, CGMH_10_wer, CGMH_10_ppl)+'\n')
        fr.write("CGMH_50 ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(CGMH_50_bertscore, CGMH_50_bleurt, CGMH_50_isacrebleu, CGMH_50_wer, CGMH_50_ppl)+'\n')    

        M2M_bertscore, M2M_bleurt, M2M_isacrebleu, M2M_wer, M2M_ppl = score(M2M, gold_sentences, input_sentences)        
        fr.write("M2M ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(M2M_bertscore, M2M_bleurt, M2M_isacrebleu, M2M_wer, M2M_ppl)+'\n')
        
        ours_bertscore, ours_bleurt, ours_isacrebleu, ours_wer, ours_ppl = score(ours, gold_sentences, input_sentences)
#         ours_v2_bertscore, ours_v2_bleurt, ours_v2_isacrebleu, ours_v2_wer, ours_v2_ppl = score(ours_v2, gold_sentences, input_sentences)
#         ours_v3_bertscore, ours_v3_bleurt, ours_v3_isacrebleu, ours_v3_wer, ours_v3_ppl = score(ours_v3, gold_sentences, input_sentences)
        fr.write("ours ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(ours_bertscore, ours_bleurt, ours_isacrebleu, ours_wer, ours_ppl)+'\n')
#         fr.write("ours_v2 ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(ours_v2_bertscore, ours_v2_bleurt, ours_v2_isacrebleu, ours_v2_wer, ours_v2_ppl)+'\n')    
#         fr.write("ours_v3 ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(ours_v3_bertscore, ours_v3_bleurt, ours_v3_isacrebleu, ours_v3_wer, ours_v3_ppl)+'\n')        
    
        overlap_v3 = count_overlap(ours_v3, input_sentences)
        overlap_one = count_overlap(ours_one, input_sentences)
        fr.write("overlap ## overlap_v3: {}, overlap_one: {}".format(overlap_v3, overlap_one)+'\n')
    
        input_bertscore, input_bleurt, input_isacrebleu, input_wer, input_ppl = score(input_sentences, gold_sentences, input_sentences)    
        ref_bertscore, ref_bleurt, ref_isacrebleu, ref_wer, ref_ppl = score(gold_sentences, gold_sentences, input_sentences)    
        fr.write("input ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(input_bertscore, input_bleurt, input_isacrebleu, input_wer, input_ppl)+'\n')
        fr.write("reference ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(ref_bertscore, ref_bleurt, ref_isacrebleu, ref_wer, ref_ppl)+'\n')    
    
    elif dataset == "medical":
        fr.write("#################### medical ####################"+'\n')
        UPSA_bertscore, UPSA_bleurt, UPSA_isacrebleu, UPSA_wer, UPSA_ppl = score(UPSA, gold_sentences, input_sentences)        
        CGMH_10_bertscore, CGMH_10_bleurt, CGMH_10_isacrebleu, CGMH_10_wer, CGMH_10_ppl = score(CGMH_10, gold_sentences, input_sentences)
        CGMH_50_bertscore, CGMH_50_bleurt, CGMH_50_isacrebleu, CGMH_50_wer, CGMH_50_ppl = score(CGMH_50, gold_sentences, input_sentences)        
        fr.write("UPSA ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(UPSA_bertscore, UPSA_bleurt, UPSA_isacrebleu, UPSA_wer, UPSA_ppl)+'\n')
        fr.write("CGMH_10 ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(CGMH_10_bertscore, CGMH_10_bleurt, CGMH_10_isacrebleu, CGMH_10_wer, CGMH_10_ppl)+'\n')
        fr.write("CGMH_50 ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(CGMH_50_bertscore, CGMH_50_bleurt, CGMH_50_isacrebleu, CGMH_50_wer, CGMH_50_ppl)+'\n')         
        ours_medical_bertscore, ours_medical_bleurt, ours_medical_isacrebleu, ours_medical_wer, ours_medical_ppl = score(ours_medical, gold_sentences, input_sentences)        
        fr.write("ours_medical ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(ours_medical_bertscore, ours_medical_bleurt, \
                                                                                                      ours_medical_isacrebleu, ours_medical_wer, ours_medical_ppl)+'\n')    
        
        input_bertscore, input_bleurt, input_isacrebleu, input_wer, input_ppl = score(input_sentences, gold_sentences, input_sentences)    
        ref_bertscore, ref_bleurt, ref_isacrebleu, ref_wer, ref_ppl = score(gold_sentences, gold_sentences, input_sentences)    
        fr.write("input ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(input_bertscore, input_bleurt, input_isacrebleu, input_wer, input_ppl)+'\n')
        fr.write("reference ## bertscore: {}, bleurt: {}, isacrebleu: {}, wer: {}, ppl: {}".format(ref_bertscore, ref_bleurt, ref_isacrebleu, ref_wer, ref_ppl)+'\n')         
    fr.close()


if __name__ == '__main__':
    main()    