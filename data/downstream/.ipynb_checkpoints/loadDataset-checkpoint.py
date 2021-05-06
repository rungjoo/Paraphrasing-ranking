from datasets import list_datasets, list_metrics, load_dataset, load_metric
datasets = list_datasets()

import random, pdb

def financial():
    """financial_phrasebank"""
    f = open('./original/financial_train.txt')
    train_lines = f.readlines()
    f.close()
    financial_train_sentence, financial_train_label = [], []
    for line in train_lines:
        sentence, label = line.strip().split('\t')
        financial_train_sentence.append(sentence)
        financial_train_label.append(label)
        
    f = open('./original/financial_test.txt')
    test_lines = f.readlines()
    f.close()    
    financial_test_sentence, financial_test_label = [], []
    for line in test_lines:
        sentence, label = line.strip().split('\t')
        financial_test_sentence.append(sentence)
        financial_test_label.append(label)

    return (financial_train_sentence, financial_train_label), (financial_test_sentence, financial_test_label)
        
def hatespeech_en():
    """hate_speech18"""
    f = open('./original/hate_speech18_train.txt')
    train_lines = f.readlines()
    f.close()
    hate_speech18_train_text, hate_speech18_train_label = [], []
    for line in train_lines:
        sentence, label = line.strip().split('\t')
        hate_speech18_train_text.append(sentence)
        hate_speech18_train_label.append(label)
        
    f = open('./original/hate_speech18_test.txt')
    test_lines = f.readlines()
    f.close()    
    hate_speech18_test_text, hate_speech18_test_label = [], []
    for line in test_lines:
        sentence, label = line.strip().split('\t')
        hate_speech18_test_text.append(sentence)
        hate_speech18_test_label.append(label)
        
    return (hate_speech18_train_text, hate_speech18_train_label), (hate_speech18_test_text, hate_speech18_test_label)

def hatespeech_kr():    
    """kor_hate"""
    f = open('./original/kor_hate_train.txt')
    train_lines = f.readlines()
    f.close()
    kor_hate_train_comments, kor_hate_train_hate = [], []
    for line in train_lines:
        sentence, label = line.strip().split('\t')
        kor_hate_train_comments.append(sentence)
        kor_hate_train_hate.append(label)
        
    f = open('./original/kor_hate_test.txt')
    test_lines = f.readlines()
    f.close()    
    kor_hate_test_comments, kor_hate_test_hate = [], []
    for line in test_lines:
        sentence, label = line.strip().split('\t')
        kor_hate_test_comments.append(sentence)
        kor_hate_test_hate.append(label)
    
    return (kor_hate_train_comments, kor_hate_train_hate), (kor_hate_test_comments, kor_hate_test_hate)
    
    