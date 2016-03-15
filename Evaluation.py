from collections import Counter
from collections import defaultdict
import nltk
from nltk import ngrams
from itertools import izip

class calculate():
        def __init__(self, ftr, fte):
                self.ftrain = ftr
                self.ftest = fte
        #Precision calculation for individual tags
        def calcPre(self, tag) :
                return float(self.truePos[tag] * 1.0 / (self.truePos[tag] + self.falsePos[tag]))
        #Recall calculation for individual tags
        def calcRec(self, tag) :
                return float(self.truePos[tag] * 1.0 / (self.truePos[tag] + self.falseNeg[tag]))
        #F1-score calculation for individual tags
        def calcF1(self,tag) :
                return 2 * float(self.pre[tag] * self.rec[tag] * 1.0 / (self.pre[tag] + self.rec[tag]))
        ##Precision calculation
        def calcPreMicro(self) :
                keys = self.truePos.values()
                sumTruePos = 0
                for key in keys :
                        sumTruePos += key
                keys = self.falsePos.values()
                sumFalsePos = 0
                for key in keys :
                        #print key
                        sumFalsePos += key
                #print sumFalsePos
                return float(sumTruePos * 1.0 / (sumTruePos + sumFalsePos))

        def calcPreMacro(self) :
                keys = self.pre.values()
                sumPre = 0
                for key in keys :
                        sumPre += key
                print len(self.tags)
                return float(sumPre * 1.0 / len(self.tags))
        #Recall calculation
        def calcRecMicro(self) :
                keys = self.truePos.values()
                sumTruePos = 0
                for key in keys :
                        sumTruePos += key
                keys = self.falseNeg.values()
                sumFalseNeg = 0
                for key in keys :
                        #print key
                        sumFalseNeg += key
                #print sumFalseNeg
                return float(sumTruePos * 1.0 / (sumTruePos + sumFalseNeg))

        def calcRecMacro(self) :
                keys = self.rec.values()
                sumRec = 0
                for key in keys :
                        sumRec += key
                return float(sumRec * 1.0 / len(self.tags))
        #F1-score calculation
        def calcF1Micro(self, preMicro, recMicro) :
                return 2 * float(preMicro * recMicro / (preMicro + recMicro)) 

        def calcF1Macro(self, preMacro, recMacro) :
                return 2 * float(preMacro * recMacro / (preMacro + recMacro))

        def calc(self):
                #self.word_tag1 = defaultdict(int)
                self.tags_total = []
                self.truePos = defaultdict(int)
                self.falsePos = defaultdict(int)
                self.falseNeg = defaultdict(int)
                self.pre = defaultdict(int)
                self.rec = defaultdict(int)
                self.f1 = defaultdict(int)
                self.confusionMarix = defaultdict(int)
                #self.tritag = defaultdict(int)
                file1 = open(self.ftrain,'r')
                file2 = open(self.ftest,'r')
	        for line1, line2 in izip(file1, file2):
	                line1 = line1.strip()
	                line2 = line2.strip()
	                l1 = line1.split(' ')
	                l2 = line2.split(' ')
	                if len(l1) == 0 : break
	                for i1, i2 in izip(l1, l2) :
	                        a1 = i1.rsplit('/',1)
	                        a2 = i2.rsplit('/',1)
                                self.confusionMarix[a1[1],a2[1]] += 1
	                        self.tags_total.append(a1[1])
	                        if a1[1] == a2[1] :
	                                self.truePos[a1[1]] += 1
	                        else :
	                                self.falsePos[a2[1]] += 1
	                                self.falseNeg[a1[1]] += 1

                self.tags  = set(self.tags_total)
                i=0
                for tag in self.tags :
                        #i = i + 1
                        #if i < 5 :
                                #print tag, self.truePos[tag] , self.falsePos[tag] , self.falseNeg[tag]
                        self.pre[tag] = self.calcPre(tag)
                        self.rec[tag] = self.calcRec(tag)
                        self.f1[tag] = self.calcF1(tag)

                preMicro = self.calcPreMicro()
                preMacro = self.calcPreMacro()
                recMicro = self.calcRecMicro()
                recMacro = self.calcRecMacro()
                f1Micro = self.calcF1Micro(preMicro, recMicro)
                f1Macro = self.calcF1Macro(preMacro, recMacro)
                print " ",
                for u in self.tags :
                        print u + " " ,
                print "\n"
                for u in self.tags :
                        print u + " " ,
                        for v in self.tags :
                                print self.confusionMarix[u,v] ,
                        print "\n"
                print "PRE Micro : " + str(preMicro) + " PRE Macro :" + str(preMacro)
                print "REC Micro : " + str(recMicro) + " REC Macro :" + str(recMacro)
                print "F1-score Micro : " + str(f1Micro) + " F1-score Macro :" + str(f1Macro)

                #self.tags = set(self.tags)

'''#############confusion matrix with laplace smoothing starts##########

      ADV    NOUN     ADP     PRON     DET     .       PRT      VERB     X    NUM     CONJ     ADJ

ADV  22650   1529     909       1       88     0       107       1000    0     0       24      1375

NOUN  1185   116486    9        2       14     0        3        2158    0     2        1      910

ADP   1390    74      63360     63      351    20      370        165    0     0        28      40

PRON  53      51       424     26371    317    0        1         12     0     0        0        0

DET  588      23       209      513    61815   0        1         4      0     0        10       1

.    641       1       47        0       0    70865     0         1      0     0        0        0

PRT  705      385     2861       2       2      0       11564     85     0     0        0        18

VERB 569      6723     68        0       0      0        8      81831    0     0        0        567

X     6       450      11        0       2      23       0       46      19    5        0        28

NUM  104      2093      0        0       75     0        0       67      0    4258      0        67

CONJ  289      5       101       0       65     0        0       4       0     0       17315      0

ADJ  1078     6469      31       0        0     0        18     1098     0     0        0       27772
        
'''


x = calculate('Data\Brown_tagged_train.txt', 'Data\output1.txt')
x.calc()
