from datasets import list_datasets, list_metrics, load_dataset, load_metric
datasets = list_datasets()

import random, pdb

def financial():
    """financial_phrasebank"""
    financial_phrasebank = load_dataset('financial_phrasebank', 'sentences_allagree')
    financial_sentence, financial_label = financial_phrasebank['train']['sentence'], financial_phrasebank['train']['label']
    
    financial_order = [x for x in range(len(financial_sentence))]
    random.shuffle(financial_order)
    train_len = int(len(financial_order)*0.9)
    financial_train_order = financial_order[:train_len]
    financial_test_order = financial_order[train_len:]
    
    financial_train_sentence, financial_train_label = [], []
    for x in financial_train_order:
        financial_train_sentence.append(financial_sentence[x])
        financial_train_label.append(financial_label[x])
        
    financial_test_sentence, financial_test_label = [], []
    for x in financial_test_order:
        financial_test_sentence.append(financial_sentence[x])
        financial_test_label.append(financial_label[x])
    
    return (financial_train_sentence, financial_train_label), (financial_test_sentence, financial_test_label)
        
def hatespeech_en():
    """hate_speech18"""
    hate_speech18 = load_dataset('hate_speech18')

    hate_speech18_text, hate_speech18_user_id, hate_speech18_subforum_id, hate_speech18_num_contexts, hate_speech18_label =\
        hate_speech18['train']['text'], hate_speech18['train']['user_id'], hate_speech18['train']['subforum_id'], hate_speech18['train']['num_contexts'], hate_speech18['train']['label']
    
    hate_speech18_order = [x for x in range(len(hate_speech18_text))]
    random.shuffle(hate_speech18_order)
    train_len = int(len(hate_speech18_order)*0.9)
    hate_speech18_train_order = hate_speech18_order[:train_len]
    hate_speech18_test_order = hate_speech18_order[train_len:]
    
    hate_speech18_train_text, hate_speech18_train_label = [], []
    for x in hate_speech18_train_order:
        hate_speech18_train_text.append(hate_speech18_text[x])
        hate_speech18_train_label.append(hate_speech18_label[x])
        
    hate_speech18_test_text, hate_speech18_test_label = [], []
    for x in hate_speech18_test_order:
        hate_speech18_test_text.append(hate_speech18_text[x])
        hate_speech18_test_label.append(hate_speech18_label[x])
        
    return (hate_speech18_train_text, hate_speech18_train_label), (hate_speech18_test_text, hate_speech18_test_label)

def hatespeech_kr():    
    """kor_hate"""
    kor_hate = load_dataset('kor_hate')

    kor_hate_train_comments, kor_hate_train_contain_gender_bias, kor_hate_train_bias, kor_hate_train_hate \
        = kor_hate['train']['comments'], kor_hate['train']['contain_gender_bias'], kor_hate['train']['bias'], kor_hate['train']['hate']

    kor_hate_test_comments, kor_hate_test_contain_gender_bias, kor_hate_test_bias, kor_hate_test_hate \
        = kor_hate['test']['comments'], kor_hate['test']['contain_gender_bias'], kor_hate['test']['bias'], kor_hate['test']['hate']
    
    return (kor_hate_train_comments, kor_hate_train_hate), (kor_hate_test_comments, kor_hate_test_hate)
    
    