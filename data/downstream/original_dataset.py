from tqdm import tqdm
import math, pdb

from downDataset import financial, hatespeech_en, hatespeech_kr

def SavePara(save_name, sentence_list, label_list):
    for src_text, label in tqdm(zip(sentence_list, label_list)):
        with open(save_name, 'a') as f:
            f.write(src_text+'\t'+str(label)+'\n')
    return

def main():
    (financial_train_sentence, financial_train_label), (financial_test_sentence, financial_test_label) = financial()
    (hate_speech18_train_text, hate_speech18_train_label), (hate_speech18_test_text, hate_speech18_test_labels) = hatespeech_en()
    (kor_hate_train_comments, kor_hate_train_hate), (kor_hate_test_comments, kor_hate_test_hate) = hatespeech_kr()
    
    SavePara('./original/financial_train.txt', financial_train_sentence, financial_train_label)
    SavePara('./original/financial_test.txt', financial_test_sentence, financial_test_label)
    
    SavePara('./original/hate_speech18_train.txt', hate_speech18_train_text, hate_speech18_train_label)
    SavePara('./original/hate_speech18_test.txt', hate_speech18_test_text, hate_speech18_test_labels)
    
    SavePara('./original/kor_hate_train.txt', kor_hate_train_comments, kor_hate_train_hate)
    SavePara('./original/kor_hate_test.txt', kor_hate_test_comments, kor_hate_test_hate)
                   
    
if __name__ == '__main__':
    main()    