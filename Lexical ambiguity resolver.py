# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 08:19:30 2019

@author: has
"""

from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec,KeyedVectors,Phrases
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
import os
from scipy import spatial
#reference="this is a test"
#print([reference.split()])
#reference = [['this', 'is', 'a', 'test']]
#candidate = ['this', 'is', 'a', 'test']
#score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#print('Cumulative 1-gram: %f' % sentence_bleu([reference.split()], candidate, weights=(1, 0, 0, 0)))
#print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
#print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
#print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
#score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
#print(score)
def Bleu_compute():
   actual, predicted = list(), list()
   dirname1="C:/Users/has/Desktop/research data/experiment 4.txt"
   with open(dirname1, encoding='utf8') as f:
      for line in f:
        words = line.rstrip('\n').split('\t')
        actual.append([words[0].split()])
        predicted.append(words[1].split())

   print(" ")    
   print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
   print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
   print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
   print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))  
  
#Bleu_compute()

def load_model():
    #am_model=KeyedVectors.load_word2vec_format('C:/Users/has/Desktop/semantic models/sampled-all-stemmed_1e-3.txt')
    #am_model=KeyedVectors.load_word2vec_format('C:/Users/has/Desktop/semantic models/all_stemmed_1e-3.txt')
    am_model=KeyedVectors.load_word2vec_format('C:/Users/Habte/Desktop/MSCS/2022/gash/final-Embedding.txt')
    
    #print(am_model.similarity('መንግስት','አገር'))
    #print(am_model.most_similar('ህዝብ',topn=10))
    #print(am_model.cosine_similarities('ወረቀት','መጸሀፍ'))
    return am_model

#load_model()
#def score(seq1, seq2, model,tail = 100, head = 10):
#	'''prepares inputs for scoring'''
#	seq1_word_list = word_tokenize(seq1.strip().lower())[-tail:]
#	seq2_word_list = word_tokenize(seq2.strip().lower())[:head]
#	return sim_score(seq1_word_list, seq2_word_list,model)

#calculating similarity between two phrases======================================================
def sim_score(wordlist1, wordlist2,model):
	'''calculates average maximum similarity between two phrase inputs'''
	maxes = []   
	for word in wordlist1:
		cur_max = 0
		for word2 in wordlist2:
			if word == word2: #case where words are identical
				sim = 1
				cur_max = sim
			elif word in model.vocab and word2 in model.vocab:	
				sim = model.similarity(word, word2) #calculate cosine similarity score for words
				if sim > cur_max:
					cur_max = sim
		if cur_max != 0:
			maxes.append(cur_max)
	if sum(maxes) == 0:
		return 0
	return float(sum(maxes)) / len(maxes)

#print("score=",sim_score("ቤተ ክርስቲያን", "ቤተ መንግስት",load_model()))

#phrase represantation====================================================================================
def phrase_representaion(phrase):
    am_model=load_model()
    ph=[]
    phra=phrase.split(" ")
    k=0
    while k<len(phra):
        ph.append(am_model[phra[k]])
        k=k+1   
    i=0
    k=0
    phrase_weight=[]
    while k<len(phra): 
       while i<300:
         phrase_weight.append((ph[k][i] + ph [k][i])/2)      
         i=i+1
       k=k+1  
    print(phrase_weight)
    
#phrase_representaion("ቤተ ክርስቲያን")
    
def tag_words_senses():
    dirname1="C:/Users/has/Desktop/research data/amharic_corpus/ambgious_words.txt"
    dirname="C:/Users/has/Desktop/research data/amharic_corpus/arranged-am-corpus"
    dirname2="C:/Users/has/Desktop/research data/amharic_corpus/all_ambigous_word_found_habtish.txt"
    
    rep=[]
    found=[]
    with open(dirname1, encoding='utf8') as f:
         for line in f:
            words = line.rstrip('\n').split('\t')
            #y=words[1]
            rep.append(words[0])               
    f.close   
              
    file=open(dirname2 , 'w', encoding='utf8')
    
    for fname in os.listdir(dirname):
        for line in open(os.path.join(dirname, fname),encoding='utf8'):
            j=0
            words=line.split()
            while j<len(rep):
                i=0
                while(i<len(words)):    
                    if(words[i]==rep[j]):
                        print("ambigues word: "+words[i])
                        #print("file name: "+ fname)
                        found.append("ambigues word: "+words[i]+"   file name: "+ fname+"\n")
                    i=i+1
                    #print("sentence that contain ambigues word: "+ line)
                j=j+1

    file.write(" ".join(found))
    file.close                        

#tag_words_senses()  

class sense_disambiguouse:
    #remove stop words from input sentence
    
    def stopword_removal(self, sentense):
        dirname1="C:/Users/Habte/Desktop/MSCS/2022/gash/all-unstemmed/habtish-data/stopwords.txt"
        removedSentence=""
        rep=[]
        with open(dirname1, encoding='utf8') as f:
             for line in f:
                words = line.rstrip('\n')
                #y=words[1]
                rep.append(words)               
        f.close
        i=0
        removedSentence=sentense
        while i<len(rep):   
            if " "+rep[i]+" " in sentense:
                removedSentence=removedSentence.replace(rep[i],"")
                #removedSentence=sentense.replace(" "+rep[i]+ "\n"," ")   
                #removedSentence=sentense.replace("\n"+ rep[i]+ " "," ")                   
            i=i+1
            removedSentence=' '.join(removedSentence.split())
        return removedSentence
    
    def disambiguate_word_detection(sentense):
        dirname1="C:/Users/Habte/Desktop/MSCS/2022/Thesis-Data/ambgious words.txt"
        ambigouseWord=""
        rep=[]
        with open(dirname1, encoding='utf8') as f:
             for line in f:
                words = line.rstrip('\n')
                #y=words[1]
                rep.append(words)               
        f.close
        
        words = sentense.split(" ")
        i=0
        while i<len(words):
            j=0
            while j<len(rep):
                if(words[i]==rep[j]):
                    ambigouseWord=rep[j]
                j=j+1
            i=i+1  
        
        return ambigouseWord
        
        
    def feature_extraction(self, ambiguateWord, sentense):
        features=[]
        words = sentense.split(" ")
        i=0
        featureCount=0
        while i<len(words):
            if(words[i] != ambiguateWord and featureCount < 10):
                features.append(words[i])
                featureCount= featureCount + 1
            i=i+1
        
        return features
    
    def compute_semantic_relatedness(self, ambiguateWord, contexts):
        #-----calculate context vector-----
        am_model=load_model()
        k=0
        ph=[]
        while k<len(contexts):
            try:
                ph.append(am_model[contexts[k]])
            except:
                ph.append([0]*300)
            k=k+1
        i=0
        k=0
        phrase_weight=[]
        while i<300:
            k=0
            vec_avg=0
            while k<len(contexts): 
                vec_avg=vec_avg + ph [k][i]      
                k=k+1
            phrase_weight.append(vec_avg/len(contexts))    
            i=i+1  
        
        #calculate semantic similarties between context vector and senses vector
        dirname1="C:/Users/Habte/Desktop/MSCS/2022/gash/all-unstemmed/Dictionary Sense.txt"
        wordsenses=[]
        senseDefination=[]
        with open(dirname1, encoding='utf8') as f:
             for line in f:
                words = line.rstrip('\n').split("\t")
                if(words[0]==ambiguateWord+"(0)" or words[0]==ambiguateWord+"(1)"):
                    wordsenses.append(words[0])
                    senseDefination.append(words[1])
        f.close
        
        cos_sim1=1 - spatial.distance.cosine(am_model[wordsenses[0]], phrase_weight)
        cos_sim2=1 - spatial.distance.cosine(am_model[wordsenses[1]], phrase_weight)
        
        if cos_sim1 > cos_sim2:
            print("The sense of the word is: "+wordsenses[0]+" and its defination is :- ", senseDefination[0])
            print("with sense score:- ", cos_sim1)
        else:
            print("The sense of the word is: "+wordsenses[1]+" and its defination is :- ", senseDefination[1])
            print("with sense score:-", cos_sim2)
        
        
    def extendedLexiDisambugation(self, ambiguateWord, contexts):
        #-----calculate context vector-----
        am_model=load_model()
        k=0
        ph=[]
        while k<len(contexts):
            try:
                ph.append(am_model[contexts[k]])
            except:
                ph.append([0]*300)
            k=k+1
        i=0
        k=0
        phrase_weight=[]
        while i<300:
            k=0
            vec_avg=0
            while k<len(contexts): 
                vec_avg=vec_avg + ph [k][i]      
                k=k+1
            phrase_weight.append(vec_avg/len(contexts))    
            i=i+1  
        
        #---Identify senses and thier defination-----
        dirname1="C:/Users/Habte/Desktop/MSCS/2022/gash/all-unstemmed/Dictionary Sense.txt"
        wordsenses=[]
        senseDefination=[]
        with open(dirname1, encoding='utf8') as f:
             for line in f:
                words = line.rstrip('\n').split("\t")
                if(words[0]==ambiguateWord+"(0)" or words[0]==ambiguateWord+"(1)"):
                    wordsenses.append(words[0])
                    senseDefination.append(words[1])
        f.close
        
        senseFeature1=[]
        senseFeature2=[]
        i=0
        while i<2:
            senseContext=senseDefination[i].split(" ")
            k=0
            if i==0:
                while k<len(senseContext):
                    senseFeature1.append(senseContext[k])
                    k=k+1
            else:
                while k<len(senseContext):
                    senseFeature2.append(senseContext[k])
                    k=k+1
            i=i+1
        
        #----calculate avarage vector for each sense
        sense_vector1=[]
        sense_vector2=[]
        avg_sense_vector1=[]
        avg_sense_vector2=[]
        k=0
        while k<len(senseFeature1):
            try:
                sense_vector1.append(am_model[senseFeature1[k]])
            except:
                sense_vector1.append([0]*300)
            k=k+1
            
        k=0
        while k<len(senseFeature2):
            try:
                sense_vector2.append(am_model[senseFeature2[k]])
            except:
                sense_vector2.append([0]*300)
            k=k+1
        
        i=0
        while i<300:
            k=0
            vec_avg=0
            while k<len(sense_vector1): 
                vec_avg=vec_avg + sense_vector1 [k][i]      
                k=k+1
            avg_sense_vector1.append(vec_avg/len(sense_vector1))    
            i=i+1
            
        i=0
        while i<300:
            k=0
            vec_avg=0
            while k<len(senseFeature2): 
                vec_avg=vec_avg + sense_vector2 [k][i]      
                k=k+1
            avg_sense_vector2.append(vec_avg/len(sense_vector2))    
            i=i+1
        
        cos_sim1=1 - spatial.distance.cosine(avg_sense_vector1, phrase_weight)
        cos_sim2=1 - spatial.distance.cosine(avg_sense_vector2, phrase_weight)
        
        if cos_sim1 > cos_sim2:
            print("The sense of the word is: "+wordsenses[0]+" and its defination is :- ", senseDefination[0])
            print("with sense score:- ", cos_sim1)
        else: 
            print("The sense of the word is: "+wordsenses[1]+" and its defination is :- ", senseDefination[1])
            print("with sense score:- ", cos_sim2)
        
    def evaluate_semantic_relatedness(self, ambiguateWord, contexts):
        #-----calculate context vector-----
        am_model=load_model()
        k=0
        ph=[]
        while k<len(contexts):
            try:
                ph.append(am_model[contexts[k]])
            except:
                ph.append([0]*300)
            k=k+1
        i=0
        k=0
        phrase_weight=[]
        while i<300:
            k=0
            vec_avg=0
            while k<len(contexts): 
                vec_avg=vec_avg + ph [k][i]      
                k=k+1
            phrase_weight.append(vec_avg/len(contexts))    
            i=i+1  
        
        #calculate semantic similarties between context vector and senses vector
        dirname1="C:/Users/Habte/Desktop/MSCS/2022/gash/all-unstemmed/Dictionary Sense.txt"
        wordsenses=[]
        senseDefination=[]
        with open(dirname1, encoding='utf8') as f:
             for line in f:
                words = line.rstrip('\n').split("\t")
                if(words[0]==ambiguateWord+"(0)" or words[0]==ambiguateWord+"(1)"):
                    wordsenses.append(words[0])
                    senseDefination.append(words[1])
        f.close
        
        cos_sim1=1 - spatial.distance.cosine(am_model[wordsenses[0]], phrase_weight)
        cos_sim2=1 - spatial.distance.cosine(am_model[wordsenses[1]], phrase_weight)
        
        if cos_sim1 > cos_sim2:
            return wordsenses[0]
        else:
            return wordsenses[1]
        
    def evaluate_LexiDisambugation(self, ambiguateWord, contexts):
        #-----calculate context vector-----
        am_model=load_model()
        k=0
        ph=[]
        while k<len(contexts):
            try:
                ph.append(am_model[contexts[k]])
            except:
                ph.append([0]*300)
            k=k+1
        i=0
        k=0
        phrase_weight=[]
        while i<300:
            k=0
            vec_avg=0
            while k<len(contexts): 
                vec_avg=vec_avg + ph [k][i]      
                k=k+1
            phrase_weight.append(vec_avg/len(contexts))    
            i=i+1  
        
        #---Identify senses and thier defination-----
        dirname1="C:/Users/Habte/Desktop/MSCS/2022/gash/all-unstemmed/Dictionary Sense.txt"
        wordsenses=[]
        senseDefination=[]
        with open(dirname1, encoding='utf8') as f:
             for line in f:
                words = line.rstrip('\n').split("\t")
                if(words[0]==ambiguateWord+"(0)" or words[0]==ambiguateWord+"(1)"):
                    wordsenses.append(words[0])
                    senseDefination.append(words[1])
        f.close
        
        senseFeature1=[]
        senseFeature2=[]
        i=0
        while i<2:
            senseContext=senseDefination[i].split(" ")
            k=0
            if i==0:
                while k<len(senseContext):
                    senseFeature1.append(senseContext[k])
                    k=k+1
            else:
                while k<len(senseContext):
                    senseFeature2.append(senseContext[k])
                    k=k+1
            i=i+1
        
        #----calculate avarage vector for each sense
        sense_vector1=[]
        sense_vector2=[]
        avg_sense_vector1=[]
        avg_sense_vector2=[]
        k=0
        while k<len(senseFeature1):
            try:
                sense_vector1.append(am_model[senseFeature1[k]])
            except:
                sense_vector1.append([0]*300)
            k=k+1
            
        k=0
        while k<len(senseFeature2):
            try:
                sense_vector2.append(am_model[senseFeature2[k]])
            except:
                sense_vector2.append([0]*300)
            k=k+1
        
        i=0
        while i<300:
            k=0
            vec_avg=0
            while k<len(sense_vector1): 
                vec_avg=vec_avg + sense_vector1 [k][i]      
                k=k+1
            avg_sense_vector1.append(vec_avg/len(sense_vector1))    
            i=i+1
            
        i=0
        while i<300:
            k=0
            vec_avg=0
            while k<len(senseFeature2): 
                vec_avg=vec_avg + sense_vector2 [k][i]      
                k=k+1
            avg_sense_vector2.append(vec_avg/len(sense_vector2))    
            i=i+1
        
        cos_sim1=1 - spatial.distance.cosine(avg_sense_vector1, phrase_weight)
        cos_sim2=1 - spatial.distance.cosine(avg_sense_vector2, phrase_weight)
        
        if cos_sim1 > cos_sim2:
            return wordsenses[0]
        else:
            return wordsenses[1]
        
sense_disambiguouse.stopword = classmethod(sense_disambiguouse.stopword_removal)
sense_disambiguouse.feature = classmethod(sense_disambiguouse.feature_extraction)
sense_disambiguouse.ambiguate_word = classmethod(sense_disambiguouse.disambiguate_word_detection)
sense_disambiguouse.semantic_relatedness = classmethod(sense_disambiguouse.compute_semantic_relatedness)
sense_disambiguouse.lexi_Disambugation = classmethod(sense_disambiguouse.extendedLexiDisambugation)
sense_disambiguouse.evaluate_supervised = classmethod(sense_disambiguouse.evaluate_semantic_relatedness)
sense_disambiguouse.evaluate_lexi_Disambugation = classmethod(sense_disambiguouse.evaluate_LexiDisambugation)

def showDemo(sentense):
    #sentense="በውሃ ውስጥ ዋና መዋኘት"  
    #sentense="ጥሩ መአዛ ያለው ልጅ ነው"
    #sentense="ያልበሰለ ለጋ ልጅ ነው"
    #sentense="ኳስን በጥፊ ለጋ"
    
    #---- sentence pre-processing for one sentence------                
    processedsentence=sense_disambiguouse.stopword(sentense)
    
    #---- Ambiguase word detection------  
    ambiguateWord=sense_disambiguouse.disambiguate_word_detection(processedsentence)
    print("the ambiguaeosu word is: "+ ambiguateWord)
    
    #---- context feature extraction------ 
    features=sense_disambiguouse.feature(ambiguateWord, processedsentence)
    print("closes context features are: ", features)
    
    #-----test the first model for one sentence-------
    sense_disambiguouse.semantic_relatedness(ambiguateWord, features)
    #-----test the second model for one sentence-------
    #sense_disambiguouse.lexi_Disambugation(ambiguateWord, features)

#showDemo("ደግ አረገ ተብሎ የሚጠላበት ሃገር")

def evalauteModel():
    #-----------Evaluation section--------
    #-----reading test file--------
    dirname1="C:/Users/Habte/Desktop/MSCS/2022/gash/all-unstemmed/check_evaluation.txt"
    eval_accuracy=0
    with open(dirname1, encoding='utf8') as f:
        for line in f:
            words = line.rstrip('\n').split("\t")
            print("sentence: "+ words[0])
            processedsentence=sense_disambiguouse.stopword(words[0])
            ambiguateWord=sense_disambiguouse.disambiguate_word_detection(processedsentence)
            print("the ambiguaeosu word is: "+ ambiguateWord)
            features=sense_disambiguouse.feature(ambiguateWord, processedsentence)
            print("closes context features are: ", features)
            sense=sense_disambiguouse.evaluate_supervised(ambiguateWord, features)
            
            if sense==words[1]:
                eval_accuracy=eval_accuracy+1
            
            
    print("The accuracy of model 1 is: ",eval_accuracy/3)
    f.close 

evalauteModel()

      

    