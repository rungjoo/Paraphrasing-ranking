from tqdm import tqdm
import math, pdb

from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
m2m_model = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M').cuda() # M2M_base
m2m_model.eval()
m2m_tokenizer = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M') # # M2M_base

m2m_model_large = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M_large').cuda()
m2m_model_large.eval()
m2m_tokenizer_large = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M_large')
print("Model Lodaing")

from generation import ParaFunc
paraphrasing = ParaFunc()

from ranking_metric import MetFunc
CalMet = MetFunc()

def generate_cands(src_text, variation=False):
    all_generations = []
#     src_text = "Should I have a hair transplant at age 24?"
    lang = 'en'
    same_smalls = paraphrasing.same_enc_dec(m2m_model, m2m_tokenizer, src_text, lang, variation=variation)
    same_larges = paraphrasing.same_enc_dec(m2m_model_large, m2m_tokenizer_large, src_text, lang, variation=variation)
    all_generations += same_smalls
    all_generations += same_larges

    tgt_langs = ['ko', 'fr', 'ja', 'zh', 'de', 'es']
    pivot_smalls = []
    pivot_larges = []
    for tgt_lang in tgt_langs:
        pivot_small = paraphrasing.pivoting(m2m_model, m2m_tokenizer, src_text, lang, tgt_lang, variation=variation)
        pivot_large = paraphrasing.pivoting(m2m_model_large, m2m_tokenizer_large, src_text, lang, tgt_lang, variation=variation)

        pivot_smalls.append(pivot_small)
        pivot_larges.append(pivot_large)
    all_generations += pivot_smalls
    all_generations += pivot_larges

    """remove overlapping"""
    overlap_generations = []
    check_chars = []
    src_chars = ''.join(src_text.split(' ')).lower()
    for generation in all_generations:
        tgt_chars = ''.join(generation.split(' ')).lower()

        if (tgt_chars != src_chars) and (tgt_chars not in check_chars):
            overlap_generations.append(generation)

        check_chars.append(tgt_chars)
        
    # max: 22 candidates (5+5+6+6)
    return overlap_generations

def ranking_cands(src_text, overlap_generations):            
    """Diversity Filtering"""
    lower_src_text = src_text.lower()
    lower_overlap_generations = [sen.lower() for sen in overlap_generations]
    
    SBlueScore_list = CalMet.CalSacreBleu(lower_src_text, lower_overlap_generations)
    WerScore_list = CalMet.CalWer(lower_src_text, lower_overlap_generations)    
    
    """v2, v3"""
    iSBlueScore_list = [-score for score in SBlueScore_list]
    if len(set(iSBlueScore_list)) == 1:
        if iSBlueScore_list[0] != 0.0:
            iSBlueScore_list = [score/iSBlueScore_list[0] for score in iSBlueScore_list]
    else:
        iSBlueScore_list = [score-min(iSBlueScore_list) for score in iSBlueScore_list]
        iSBlueScore_list = [score/max(iSBlueScore_list) for score in iSBlueScore_list]
    
    if len(set(WerScore_list)) == 1:
        if WerScore_list[0] != 0.0:
            WerScore_list = [score/WerScore_list[0] for score in WerScore_list]
    else:
        WerScore_list = [score-min(WerScore_list) for score in WerScore_list]
        WerScore_list = [score/max(WerScore_list) for score in WerScore_list]

    diversity_score_list = []
    for isblue_score, wer_score in zip(iSBlueScore_list, WerScore_list):
        diversity_score = isblue_score + wer_score
        diversity_score_list.append(diversity_score)        
    """v2, v3"""    
    
    """v1"""        
#     SBlueScore_list = [score/max(SBlueScore_list) for score in SBlueScore_list]
#     WerScore_list = [score/max(WerScore_list) for score in WerScore_list]

#     diversity_score_list = []
#     for sblue_score, wer_score in zip(SBlueScore_list, WerScore_list):
#         diversity_score = 1-sblue_score + wer_score
#         diversity_score_list.append(diversity_score)
    """v1"""

    ## remove lower diversity (max 5)
    diversity_threshold = sorted(diversity_score_list, reverse=True)[min(5, math.floor(len(diversity_score_list)/2))]
    diversity_generations = []
    for generation, score in zip(overlap_generations, diversity_score_list):
        if score >= diversity_threshold:
            diversity_generations.append(generation)
            
    """Fluency Filtering"""
    lower_diversity_generation = [sen.lower() for sen in diversity_generations]
    PPL_list = CalMet.CalPPL(lower_diversity_generation)
    
    ## remain larger PPL (max 3)
    fluency_threshold = sorted(PPL_list, reverse=False)[min(3, math.floor(len(PPL_list)/2))]
    fluency_generations = []
    for PPL, generation in zip(PPL_list, diversity_generations):
        if PPL <= fluency_threshold:
            fluency_generations.append(generation)
            
    """Semantically Filtering"""
    lower_fluency_generations = [sen.lower() for sen in fluency_generations]

    Bertscore_list = CalMet.CalBertScore(lower_src_text, lower_fluency_generations)
    BleurtScore_list = CalMet.CalBleurt(lower_src_text, lower_fluency_generations)
    
    """v2, v3"""
    if len(set(Bertscore_list)) == 1:
        Bertscore_list = [score/Bertscore_list[0] for score in Bertscore_list]
    else:
        Bertscore_list = [score-min(Bertscore_list) for score in Bertscore_list]
        Bertscore_list = [score/max(Bertscore_list) for score in Bertscore_list]   

    if len(set(BleurtScore_list)) == 1:
        BleurtScore_list = [score/BleurtScore_list[0] for score in BleurtScore_list]
    else:        
        BleurtScore_list = [score-min(BleurtScore_list) for score in BleurtScore_list] 
        BleurtScore_list = [score/max(BleurtScore_list) for score in BleurtScore_list]        
    """v2, v3"""
    
    """v1"""
#     Bertscore_list = [score/max(Bertscore_list) for score in Bertscore_list]
#     BleurtScore_list = [score/max(BleurtScore_list) for score in BleurtScore_list]   
    """v1"""
    
    semantic_score_list = []
    for i, (bert_score, bleurt_score) in enumerate(zip(Bertscore_list, BleurtScore_list)):
        semantic_score = bert_score + bleurt_score
        semantic_score_list.append(semantic_score)
        
    ## best semantic select
    max_value = sorted(semantic_score_list, reverse=True)[0]
    max_ind = semantic_score_list.index(max_value)
    final_generation = fluency_generations[max_ind]
    
    return final_generation

def main():
    dataset = "medical" # QQP
    print('dataset: ', dataset)
    
    if dataset == "QQP":
        save_path = './data/results/ours_v3.txt'
        f = open('./data/QQP_test_not_ref.txt')
        testset = f.readlines()
        f.close()
        testset = [x.strip() for x in testset]        
    elif dataset == "medical":
        save_path = './data/results/ours_medical.txt' # _v4
        f = open('./data/medical.txt')
        testline = f.readlines()
        f.close()
        testset = []
        for line in testline:
            q1 = line.split('\t')[0].strip()
            testset.append(q1)
    
#     for src_text in tqdm(testset):
    for x in tqdm(range(0, len(testset))):
        src_text = testset[x]
        overlap_generations = generate_cands(src_text)
        if len(overlap_generations) < 3: # (v2: == 0)
            overlap_generations += generate_cands(src_text, variation=True)
                
        if len(overlap_generations) == 0:
            final_generation = src_text
        else:
            final_generation = ranking_cands(src_text, overlap_generations)
        with open(save_path, 'a') as fo:
            fo.write(final_generation+'\n')
        
        """Grammar Filtering"""
#         grammar_sentence = final_generation
#         for _ in range(3):
#             matches = CalMet.grammar_metric.check(grammar_sentence)
#             if len(matches) == 0:
#                 break
#             error_fix_text = ''
#             try:
#                 match = matches[0]
#                 if 'possible' in match.message.lower():
#                     break
#                 words, offset, length = match.replacements, match.offset, match.errorLength    
#                 error_fix_text += grammar_sentence[:offset]
#                 error_fix_text += words[0]
#                 error_fix_text += grammar_sentence[offset+length:]
#             except:
#                 break                    
#             grammar_sentence = error_fix_text
#         if final_generation != grammar_sentence:
#             with open('./data/results/ours_grammar_check.txt', 'a') as fc:
#                 fc.write(src_text+'\t'+final_generation+'\t'+grammar_sentence+'\n')
#         with open('./data/results/ours_grammar.txt', 'a') as fg:
#             fg.write(grammar_sentence+'\n')
    
if __name__ == '__main__':
    main()    