from tqdm import tqdm
import math, pdb

from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
m2m_model = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M').cuda() # M2M_large
m2m_model.eval()
m2m_tokenizer = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M') # # M2M_large

m2m_model_large = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M_large').cuda()
m2m_model_large.eval()
m2m_tokenizer_large = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M_large')
print("Model Lodaing")

from m2m_generation import ParaFunc
paraphrasing = ParaFunc()

from ranking_metric import MetFunc, MetFuncKo
CalMet = MetFunc()
CalMetKo = MetFuncKo()

from loadDataset import financial, hatespeech_en, hatespeech_kr

def generate_cands(src_text, variation=False, lang='en'):
    src_lang = lang
    all_generations = []
    same_smalls = paraphrasing.same_enc_dec(m2m_model, m2m_tokenizer, src_text, src_lang, variation=variation)
    same_larges = paraphrasing.same_enc_dec(m2m_model_large, m2m_tokenizer_large, src_text, src_lang, variation=variation)
    all_generations += same_smalls
    all_generations += same_larges

    if src_lang == 'ko':
        tgt_langs = ['en', 'fr', 'ja', 'zh', 'de', 'es']
    elif src_lang == 'en':
        tgt_langs = ['ko', 'fr', 'ja', 'zh', 'de', 'es']
        
    pivot_smalls = []
    pivot_larges = []
    for tgt_lang in tgt_langs:
        pivot_small = paraphrasing.pivoting(m2m_model, m2m_tokenizer, src_text, src_lang, tgt_lang, variation=variation)
        pivot_large = paraphrasing.pivoting(m2m_model_large, m2m_tokenizer_large, src_text, src_lang, tgt_lang, variation=variation)

        pivot_smalls.append(pivot_small)
        pivot_larges.append(pivot_large)
    all_generations += pivot_smalls
    all_generations += pivot_larges

    """중복 제거"""
    overlap_generations = []
    check_chars = []
    src_chars = ''.join(src_text.split(' ')).lower()
    for generation in all_generations:
        tgt_chars = ''.join(generation.split(' ')).lower()

        if (tgt_chars != src_chars) and (tgt_chars not in check_chars):
            overlap_generations.append(generation)

        check_chars.append(tgt_chars)
        
    # 최대: 22개 후보 (5+5+6+6)
    return overlap_generations

def ranking_cands(src_text, overlap_generations):            
    """Diversity 필터링"""
    lower_src_text = src_text.lower()
    lower_overlap_generations = [sen.lower() for sen in overlap_generations]
    
    SBlueScore_list = CalMet.CalSacreBleu(lower_src_text, lower_overlap_generations)
    WerScore_list = CalMet.CalWer(lower_src_text, lower_overlap_generations)
    
    """v2, v3"""
    iSBlueScore_list = [-score for score in SBlueScore_list]
    if len(set(iSBlueScore_list)) == 1:
        iSBlueScore_list = [1.0 for _ in iSBlueScore_list]
    else:
        iSBlueScore_list = [score-min(iSBlueScore_list) for score in iSBlueScore_list]
        iSBlueScore_list = [score/max(iSBlueScore_list) for score in iSBlueScore_list]
    
    if len(set(WerScore_list)) == 1:
        WerScore_list = [1.0 for _ in WerScore_list]
    else:
        WerScore_list = [score-min(WerScore_list) for score in WerScore_list]
        WerScore_list = [score/max(WerScore_list) for score in WerScore_list]    

    diversity_score_list = []
    for isblue_score, wer_score in zip(iSBlueScore_list, WerScore_list):
        diversity_score = isblue_score + wer_score
        diversity_score_list.append(diversity_score)
    """v2, v3"""

    ## 다양성이 낮은 것은 없애기, 최대 5개 남기기    
    diversity_threshold = sorted(diversity_score_list, reverse=True)[min(5, math.floor(len(diversity_score_list)/2))]
    diversity_generations = []
    for generation, score in zip(overlap_generations, diversity_score_list):
        if score >= diversity_threshold:
            diversity_generations.append(generation)
            
    """Fluency 필터링"""
    lower_diversity_generation = [sen.lower() for sen in diversity_generations]
    PPL_list = CalMet.CalPPL(lower_diversity_generation)
        
    filter_index = [] ## nan filtering
    for i, ppl in enumerate(PPL_list):
        if not math.isnan(ppl):
            filter_index.append(i)    

    filter_PPL_list = []
    filter_diversity_generations = []
    for i in filter_index:
        filter_PPL_list.append(PPL_list[i])
        filter_diversity_generations.append(diversity_generations[i])

    ## 최대 3개 남기기
    fluency_threshold = sorted(filter_PPL_list, reverse=False)[min(3, math.floor(len(filter_PPL_list)/2))]
    fluency_generations = []
    for PPL, generation in zip(filter_PPL_list, filter_diversity_generations):
        if PPL <= fluency_threshold:
            fluency_generations.append(generation)
            
    """Semantically 필터링"""
    lower_fluency_generations = [sen.lower() for sen in fluency_generations]

    Bertscore_list = CalMet.CalBertScore(lower_src_text, lower_fluency_generations)
    BleurtScore_list = CalMet.CalBleurt(lower_src_text, lower_fluency_generations)
    
    """v2, v3"""
    if len(set(Bertscore_list)) == 1:
        Bertscore_list = [1.0 for _ in Bertscore_list]
    else:
        Bertscore_list = [score-min(Bertscore_list) for score in Bertscore_list]
        Bertscore_list = [score/max(Bertscore_list) for score in Bertscore_list]   

    if len(set(BleurtScore_list)) == 1:
        BleurtScore_list = [score/BleurtScore_list[0] for score in BleurtScore_list]
    else:        
        BleurtScore_list = [score-min(BleurtScore_list) for score in BleurtScore_list] 
        BleurtScore_list = [score/max(BleurtScore_list) for score in BleurtScore_list]        
    """v2, v3"""    
    
    semantic_score_list = []
    for i, (bert_score, bleurt_score) in enumerate(zip(Bertscore_list, BleurtScore_list)):
        semantic_score = bert_score + bleurt_score
        semantic_score_list.append(semantic_score)
        
    ## 가장 큰 semantic 고르기
    max_value = sorted(semantic_score_list, reverse=True)[0]
    max_ind = semantic_score_list.index(max_value)
    final_generation = fluency_generations[max_ind]
    
    return final_generation

def ranking_cands_ko(src_text, overlap_generations):            
    """Diversity 필터링"""
    lower_src_text = src_text.lower()
    lower_overlap_generations = [sen.lower() for sen in overlap_generations]
    
    SBlueScore_list = CalMetKo.CalSacreBleu(lower_src_text, lower_overlap_generations)
    WerScore_list = CalMetKo.CalWer(lower_src_text, lower_overlap_generations)
    
    """v2, v3"""
    iSBlueScore_list = [-score for score in SBlueScore_list]
    if len(set(iSBlueScore_list)) == 1:
        iSBlueScore_list = [1.0 for _ in iSBlueScore_list]
    else:
        iSBlueScore_list = [score-min(iSBlueScore_list) for score in iSBlueScore_list]
        iSBlueScore_list = [score/max(iSBlueScore_list) for score in iSBlueScore_list]
    
    if len(set(WerScore_list)) == 1:
        WerScore_list = [1.0 for _ in WerScore_list]
    else:
        WerScore_list = [score-min(WerScore_list) for score in WerScore_list]
        WerScore_list = [score/max(WerScore_list) for score in WerScore_list]    

    diversity_score_list = []
    for isblue_score, wer_score in zip(iSBlueScore_list, WerScore_list):
        diversity_score = isblue_score + wer_score
        diversity_score_list.append(diversity_score)
    """v2, v3"""

    ## 다양성이 낮은 것은 없애기, 최대 5개 남기기    
    diversity_threshold = sorted(diversity_score_list, reverse=True)[min(5, math.floor(len(diversity_score_list)/2))]
    diversity_generations = []
    for generation, score in zip(overlap_generations, diversity_score_list):
        if score >= diversity_threshold:
            diversity_generations.append(generation)
            
    """Fluency 필터링"""
    lower_diversity_generation = [sen.lower() for sen in diversity_generations]
    PPL_list = CalMetKo.CalPPL(lower_diversity_generation)
        
    filter_index = [] ## nan filtering
    for i, ppl in enumerate(PPL_list):
        if not math.isnan(ppl):
            filter_index.append(i)    

    filter_PPL_list = []
    filter_diversity_generations = []
    for i in filter_index:
        filter_PPL_list.append(PPL_list[i])
        filter_diversity_generations.append(diversity_generations[i])

    ## 최대 3개 남기기
    fluency_threshold = sorted(filter_PPL_list, reverse=False)[min(3, math.floor(len(filter_PPL_list)/2))]
    fluency_generations = []
    for PPL, generation in zip(filter_PPL_list, filter_diversity_generations):
        if PPL <= fluency_threshold:
            fluency_generations.append(generation)
            
    """Semantically 필터링"""
    lower_fluency_generations = [sen.lower() for sen in fluency_generations]
    Bertscore_list = CalMetKo.CalBertScore(lower_src_text, lower_fluency_generations)
    
    """v2, v3"""
    if len(set(Bertscore_list)) == 1:
        Bertscore_list = [1.0 for _ in Bertscore_list]
    else:
        Bertscore_list = [score-min(Bertscore_list) for score in Bertscore_list]
        Bertscore_list = [score/max(Bertscore_list) for score in Bertscore_list]
    """v2, v3"""    
    
    semantic_score_list = Bertscore_list
        
    ## 가장 큰 semantic 고르기
    max_value = sorted(semantic_score_list, reverse=True)[0]
    max_ind = semantic_score_list.index(max_value)
    final_generation = fluency_generations[max_ind]
    
    return final_generation

def SavePara(save_name, sentence_list, label_list, src_lang):
    for i, (src_text, label) in tqdm(enumerate(zip(sentence_list, label_list))):
        try:
            overlap_generations = generate_cands(src_text, variation=False, lang=src_lang)
            if len(overlap_generations) == 0: # (v2: == 0)
                overlap_generations += generate_cands(src_text, variation=True, lang=src_lang)
            overlap_generations = [x.strip() for x in overlap_generations if len(x.strip())>0]
            
            if len(overlap_generations) == 0:
                final_generation = src_text
            elif src_lang == 'en':
                final_generation = ranking_cands(src_text, overlap_generations)
            elif src_lang == 'ko':
                final_generation = ranking_cands_ko(src_text, overlap_generations)
            else:
                print('Error')
                break
        except Exception as e:
            print(e)
            print(src_text)
            pdb.set_trace()
            overlap_generations = generate_cands(src_text, variation=False, lang=src_lang)
            final_generation = ranking_cands(src_text, overlap_generations)

        with open(save_name, 'a') as f:
            f.write(src_text+'\t'+final_generation+'\t'+str(label)+'\n')

    return

def main():
    (financial_train_sentence, financial_train_label), (financial_test_sentence, financial_test_label) = financial()
    (hate_speech18_train_text, hate_speech18_train_label), (hate_speech18_test_text, hate_speech18_test_labels) = hatespeech_en()
    (kor_hate_train_comments, kor_hate_train_hate), (kor_hate_test_comments, kor_hate_test_hate) = hatespeech_kr()
    
    """transformers==4.5.0"""
#     SavePara('./paraphrase/financial_train.txt', financial_train_sentence, financial_train_label, src_lang='en')
#     SavePara('./paraphrase/financial_test.txt', financial_test_sentence, financial_test_label, src_lang='en')
    
#     SavePara('./paraphrase/hate_speech18_train.txt', hate_speech18_train_text, hate_speech18_train_label, src_lang='en')
#     SavePara('./paraphrase/hate_speech18_test.txt', hate_speech18_test_text, hate_speech18_test_labels, src_lang='en')
    
#     SavePara('./paraphrase/kor_hate_train.txt', kor_hate_train_comments, kor_hate_train_hate, src_lang='ko')
    SavePara('./paraphrase/kor_hate_test.txt', kor_hate_test_comments, kor_hate_test_hate, src_lang='ko')
                   
    
if __name__ == '__main__':
    main()    