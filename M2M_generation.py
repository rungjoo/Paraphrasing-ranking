from tqdm import tqdm
import math, pdb

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

def generate_one(src_text):
    lang = 'en'
#     same_smalls = paraphrasing.same_enc_dec(m2m_model, m2m_tokenizer, src_text, lang, one_generation=True)
    same_larges = paraphrasing.same_enc_dec(m2m_model_large, m2m_tokenizer_large, src_text, lang, one_generation=True)
    
    return same_larges[0]

def main():    
#     f = open('./data/QQP_test_not_ref.txt')
#     testset = f.readlines()
#     f.close()
#     testset = [x.strip() for x in testset]
    
    f = open('./data/medical.txt')
    testset_line = f.readlines()
    f.close()
    testset = []
    for line in testset_line:
        testset.append(line.split('\t')[0].strip())
    
    for x in tqdm(range(0, len(testset))):
        src_text = testset[x]
        generation = generate_one(src_text)
        with open('./data/results/M2M/M2M.txt', 'a') as fo:
#         with open('./data/results/M2M/M2M_base_medical.txt', 'a') as fo:        
            fo.write(generation+'\n')
    
if __name__ == '__main__':
    main()