import xml.etree.ElementTree as ET
import nltk
import string
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
import csv

def find_pun(s):

    #remove stopword
    clean_string = []
    for word in s:
        if word.lower() not in nltk_stopwords and word.lower() not in string.punctuation:
            clean_string.append(word.lower())
    
    # use lesk find sense
    lesk_list = []
    for word in clean_string:
        try:
            if lesk(clean_string, word) is not None:
                lesk_list.append(lesk(clean_string, word))
            else:
                lesk_list.append(0)

        except:
            lesk_list.append(0)
    
    #把逐個換成oov，再用lesk，有變的就是pun
    for i in range(len(clean_string)):
        oov_lesk_list = []
        oov_string = clean_string[:]
        oov_string[i] = 'asdasdas'
        for word in clean_string:
            try:
                oov_lesk_list.append(lesk(oov_string, word))
            except:
                oov_lesk_list.append(0)
            
        for j in range(len(lesk_list)):
            if oov_lesk_list[j] != lesk_list[j] and i!= j:
                return clean_string[j]
    
    #找別的sense中跟其他word最高的similarity，最高的就是
    max_similarity = 0
    max_word = ''
    for i in range(len(clean_string)):
        for j in range(len(clean_string)):
            synsets = wn.synsets(clean_string[i])
            for s in synsets:
                try:
                    if s.wup_similarity(lesk_list[j]) > max_similarity and i!=j:
                        if s != lesk_list[i]:
                            max_similarity =  s.wup_similarity(lesk_list[j])
                            max_word = clean_string[i]
                except:
                    print(lesk_list)
                    input()
    

    return(max_word)







if __name__ == '__main__':
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    tree = ET.parse(r'data\testing_set\data_homo_test.xml')
    root = tree.getroot()

    with open('sub.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text_id','word_id'])
        for s in root:
            st = []
            word2ID = {}
            for w in s:
                st.append(w.text.lower())
                word2ID[w.text.lower()] = w.attrib['id']
            
            word = find_pun(st)
            writer.writerow([s.attrib['id'],word2ID[word]])