from tqdm import tqdm
import math, pdb
import os

from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer
m2m_model = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M').cuda() # M2M_base
m2m_model.eval()
m2m_tokenizer = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M') # M2M_base

m2m_model_large = M2M100ForConditionalGeneration.from_pretrained('/data/private/transformer/M2M_large').cuda()
m2m_model_large.eval()
m2m_tokenizer_large = M2M100Tokenizer.from_pretrained('/data/private/transformer/M2M_large')
print("Model Lodaing")

from generation import ParaFunc
paraphrasing = ParaFunc(num_beams=1, no_repeat_ngram_size=3, num_return_sequences=1)

# from loadDataset import financial, hatespeech_en, hatespeech_kr

def SavePara(data_path):
    f = open(data_path, 'r')
    dataset = f.readlines()
    f.close()
    for line in tqdm(dataset):
        src_text, _, label = line.strip().split('\t')
        final_generation = generate_one(src_text)
        
        dirname, basename = os.path.split(data_path)
        basename = basename.split('.')[0]+'_m2m.txt'
        with open(os.path.join(dirname, basename), 'a') as fw:
            fw.write(src_text+'\t'+final_generation+'\t'+str(label)+'\n')
            

def generate_one(src_text):
    lang = 'en'
#     same_smalls = paraphrasing.same_enc_dec(m2m_model, m2m_tokenizer, src_text, lang, one_generation=True)
    same_larges = paraphrasing.same_enc_dec(m2m_model_large, m2m_tokenizer_large, src_text, lang, one_generation=True)
    
    return same_larges[0]

def main():    
    """financial"""
    train_path = './paraphrase/financial/financial_train_split.txt'
    dev_path = './paraphrase/financial/financial_dev_split.txt'
    test_path = './paraphrase/financial/financial_test.txt'
    SavePara(train_path)
    SavePara(dev_path)
    SavePara(test_path)
    
    """hate_speech18 eng"""    
    train_path = './paraphrase/hate_speech18/hate_speech18_train_split_norm.txt'
    dev_path = './paraphrase/hate_speech18/hate_speech18_dev_split_norm.txt'
    test_path = './paraphrase/hate_speech18/hate_speech18_test_split_norm.txt'
    SavePara(train_path)
    SavePara(dev_path)
    SavePara(test_path)
    
    """hate_speech kor"""
    train_path = './paraphrase/kor_hate/kor_hate_train_split.txt'
    dev_path = './paraphrase/kor_hate/kor_hate_dev_split.txt'
    test_path = './paraphrase/kor_hate/kor_hate_test.txt'  
    SavePara(train_path)
    SavePara(dev_path)
    SavePara(test_path)
    
if __name__ == '__main__':
    main()