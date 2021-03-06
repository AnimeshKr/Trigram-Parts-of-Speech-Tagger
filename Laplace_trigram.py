from collections import Counter
from collections import defaultdict
import nltk
from nltk import ngrams

class HMM():
        def __init__(self, ftr, fte):
                self.ftrain = ftr
                self.ftest = fte

        #### Function to calculate word, word-tag pair, trigram,  bigram, unigram counts ######
        def count(self):
                self.word_tag1 = defaultdict(int)
                self.tags_total = []
                #### using defaultdict to initialize the dictionaries with count = 0 ####
                self.words = defaultdict(int)
                self.unitag = defaultdict(int)
                self.bitag = defaultdict(int)
                self.tritag = defaultdict(int)
                ## going through the training file line by line to get word, word_tag count ##
                for line in open(self.ftrain,'r'):
                        line = line.strip()
                        l = line.split(' ')
                        if len(l) == 0 : break
                        ## appending two '*' before each sentence to take care of inital ngrams of the sentence
                        self.tags_total.append('*')
                        self.tags_total.append('*')
                        for i in l:
                                a = i.rsplit('/',1)
                                self.tags_total.append(a[1])
                                self.words[a[0]] +=1
                                self.word_tag1[(a[0],a[1])] += 1

                self.tags  = set(self.tags_total)
                self.tags.remove('*')
                self.word_tag = defaultdict(int)

                ## taking care of word,tag frequency < 6 ##
                for (w,t) in self.word_tag1:
                        if self.words[w]<6 :
                                self.word_tag[("rare",t)] += self.word_tag1[(w,t)]
                        else:
                                self.word_tag[(w,t)] += self.word_tag1[(w,t)]
                ## took care ##

                #### using nltk ngrams to calculate counts of unigram, bigram, trigram tags ####
                self.unit = ngrams(self.tags_total,1)
                self.unitag = Counter(self.unit)
                self.bit = ngrams(self.tags_total,2)
                self.bitag = Counter(self.bit)
                self.trit = ngrams(self.tags_total,3)
                self.tritag = Counter(self.trit)
                

        def calculate_q(self,tag1,tag2,tag3):
                return float(self.tritag[(tag1,tag2,tag3)]+1.0)/(self.bitag[(tag1,tag2)]+len(self.words))

        def calculate_e(self,word,tag):
                #return float(self.word_tag[(word,tag)]+1.0)/(self.unitag[(tag,)])
                return float(self.word_tag[(word,tag)]+1.0)/(self.unitag[(tag,)]+len(self.words))

        ##### Calcuate the parameters like q and e for HMM #####
        def calculate_params(self):
                self.e = defaultdict(int)
                self.q = defaultdict(int)
                for (w,t) in self.word_tag :
                        self.e[(w,t)]= self.calculate_e(w,t)
                for (t1,t2,t3) in self.tritag :
                        self.q[(t1,t2,t3)]= self.calculate_q(t1,t2,t3)

        #### A module to print the word/tag for each sentence in the output file ####
        def printmodule(self,sent,t,opf):
                for i in range(len(sent)):
                        opf.write(sent[i]+'/'+t[i]+' ')
                opf.write('\n')

        ### A helper function for running the Viterbi algorithm ###
        def viterbiUtil(self,opfile):
                self.count()
                self.calculate_params()
                opf= open(opfile,'w')
                f= open(self.ftest,'r')
                ind=0
                for line in f:
                        sent = line.split()
                        t=self.viterbi(sent)
                        self.printmodule(sent,t,opf)
                       # ind+=1
                       # if ind > 10 : break
                opf.close()

        #### The Vitebi Algorithm #####
        def viterbi(self,sentence):
                Pi = {}
                Bp = {}
                Pi[(-1,'*','*')] = 1 ## Base case ##
                S = {} ## contains the possible tags for a sentence position ##
                S['-2'], S['-1'] = ['*'], ['*']  
                for i in range(len(sentence)):
                        S[str(i)]=self.tags
                for k in range(len(sentence)):
                        if self.words[sentence[k]] < 6:  ## Taking care of rare words in training set ##
                                word_i="rare"
                        else:
                                word_i = sentence[k]
                        for u in S[str(k-1)]:
                                for v in S[str(k)]:
                                        mx,w1 = -1,""
                                        for w in S[str(k-2)]:
                                                if Pi[(k-1,w,u)]*self.q[(w,u,v)]*self.e[(word_i,v)] > mx:
                                                        mx = Pi[(k-1,w,u)]*self.q[(w,u,v)]*self.e[(word_i,v)]
                                                        w1 = w
                                        Pi[(k,u,v)],Bp[(k,u,v)] =  mx, w1
                                        
                n = len(sentence)
                Y = ["i"]*n
                dum=-1
                for u in self.tags:
                        for v in self.tags:
                                if Pi[(n-1,u,v)] > dum :
                                        dum,Y[n-2],Y[n-1]=Pi[(n-1,u,v)], u, v
                for k in range(n-3,-1,-1):
                        Y[k] = Bp[k+2,Y[k+1],Y[k+2]]
                return Y ## Returning tag sequence for the sentence ##

x = HMM('Data/Brown_tagged_train.txt', 'Data/Brown_train.txt')
x.viterbiUtil('Data/output1.txt')

