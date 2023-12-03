import numpy as np
import pandas as pd
import re
import string
import nltk
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

########
#PART A#
########
def getClassProbs(classData):
    # returns natural log-probabilities for numerical stability

    classMap = {j:i for i,j in enumerate(classData.unique())}
    invClassMap = {i:j for i,j in enumerate(classData.unique())}
    count = defaultdict(lambda : 0)
    for i in classData:
        count[i] +=1
    classProbs = {}
    for i in list(classMap.keys()):
        #calculating probabilities using Laplace Add-1 Smoothing
        classProbs[classMap[i]] = np.log(count[i]+1) - np.log(sum(count.values())+len(count.keys()))
    return classProbs, classMap, invClassMap

#building n-gram model
def getNgrams(inText : str, n : int):
    if n == 1:
        return inText.split()
    nGrams = []
    inText = inText.split()
    for i in range(0,len(inText)-(n-1)):
        nGrams.append(tuple(inText[i:i+n]))
    return nGrams
    
def getCounts(train,cMap,n = 1):
    mainCount = defaultdict(lambda : defaultdict(lambda : 0)) # counts prob wrd|class
    allCount = defaultdict(lambda : 0) # keeps a count of all words occuring to get vocab size
    denom = defaultdict(lambda : 0)
    for i,j in zip(train['Sentiment'],train['CoronaTweet']):
        splitt = getNgrams(j,n)
        for wrd in splitt:
            mainCount[cMap[i]][wrd] += 1 #not adding +1 for smoothing. At prediction time add +1 to all prediction such that 0 occ also get 1
            allCount[wrd] +=1
        denom[cMap[i]] += len(splitt)
    vocabSize = len(list(allCount.keys()))
    #adding |V| for laplace smoothing
    for i in list(denom.keys()):
        denom[i]+= vocabSize
    
    return mainCount, denom

def getPredProbClass(inText: list, k : int, count : defaultdict, denom : defaultdict):
    textProb = 0
    for i in inText:
        textProb += np.log(count[k][i] + 1)- np.log(denom[k])
    return textProb

def predict(inText : str, nGrams : list, cProbs : dict , icMap : dict):
    # count is a list of tuples : (n, count, denom)
    probs = []
    for i in list(cProbs.keys()):
        prob = 0
        for gram in nGrams:
            prob += getPredProbClass(getNgrams(inText,gram[0]), i, gram[1], gram[2])
        probs.append(prob + cProbs[i])
    return icMap[np.argmax(np.array(probs))],probs

trainSourceRaw = pd.read_csv('data/Corona_train.csv')
testSourceRaw = pd.read_csv("data/Corona_validation.csv")
cProbs, cMap, icMap = getClassProbs(trainSourceRaw["Sentiment"])
mainCountRawUni, denomUni = getCounts(trainSourceRaw,cMap)

matches = 0
for i,j in zip(trainSourceRaw["Sentiment"],trainSourceRaw["CoronaTweet"]):
    if i == predict(j,[(1,mainCountRawUni,denomUni)],cProbs,icMap)[0] :
        matches+=1
print(f'Part(a) : Accuracy over training data : {matches/len(trainSourceRaw)}')

matches = 0
parta_ValPreds = [predict(j,[(1,mainCountRawUni,denomUni)],cProbs,icMap)[0] for j in testSourceRaw['CoronaTweet']]
for i,j in zip(testSourceRaw["Sentiment"],parta_ValPreds):
    if i == j :
        matches+=1
partA_val = matches/len(testSourceRaw)
print(f'Part(a) : Accuracy over validation data : {matches/len(testSourceRaw)}')

#generating WordCloud
def drawWrdCld(df,path):
   wordcloud = WordCloud(background_color='white')
   wordcloud.generate_from_frequencies(frequencies=df)
   plt.figure(figsize = (20,15))
   plt.imshow(wordcloud, interpolation="bilinear")
   plt.axis("off")
   plt.savefig(path)
   plt.show()
# drawWrdCld(mainCountRawUni[0],'plots/partA_positive_wrdCloud.png')
# drawWrdCld(mainCountRawUni[1],'plots/partA_negative_wrdCloud.png')
# drawWrdCld(mainCountRawUni[2],'plots/partA_neutral_wrdCloud.png')

########
#PART B#
########
#validation set accuracy by randomly guessing
matches = 0
partB_predsRandom = [icMap[i] for i in np.random.randint(0,3,len(testSourceRaw))]
for i,j in zip(testSourceRaw["Sentiment"],partB_predsRandom):
    if i == j :
        matches+=1
partB_random = matches/len(testSourceRaw)
print(f'Part(b) : Validation accuracy by randomly predicting : {matches/len(testSourceRaw)}')

#validation accuracy by predicting each example positive
matches = 0
partB_predsPositive = ['Positive']*len(testSourceRaw)
for i in testSourceRaw["Sentiment"]:
    if i == "Positive" :
        matches+=1
partB_pos = matches/len(testSourceRaw)
print(f'Part(b) : Validation accuracy by predicting positive : {matches/len(testSourceRaw)}')
#improvements
print(f'Part(b) : Naive Bayes outperforms :\n1) Random Baseline by {partA_val - partB_random}\n1) Positive Baseline by {partA_val - partB_pos}\nIn terms of accuracy over validation dataset')

########
#PART C#
########

cm = confusion_matrix(list(testSourceRaw['Sentiment']),parta_ValPreds)
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral','Positive'], 
                     columns = ['Negative','Neutral','Positive'],dtype=np.int32)
print("The confusion matrix for the given task : ")
print(cm_df)

cm = confusion_matrix(list(testSourceRaw['Sentiment']),partB_predsRandom)
cm_df = pd.DataFrame(cm,
                     index = ['Negative','Neutral','Positive'], 
                     columns = ['Negative','Neutral','Positive'],dtype=np.int32)
print("The confusion matrix for random prediction : ")
print(cm_df)

########
#PART D#
########

class wordEdit :

    def __init__(self,stmLem = 1):
        if stmLem == 0:
            self.editor = None
        elif stmLem == 1 :
            self.editor = PorterStemmer()
        else :
            self.editor = WordNetLemmatizer()
        self.stmLem = stmLem

    def __call__(self, s : str):
        if self.stmLem == 0:
            return s
        elif self.stmLem == 1:
            return self.editor.stem(s)
        else:
            return self.editor.lemmatize(s)

def processInput(inText : str, wrdEdit : wordEdit,stopwords, keePuncs = 0):
    if keePuncs == 0:
        stemmed = [wrdEdit(i) for i in nltk.tokenize.word_tokenize(re.sub('[{}]'.format(string.punctuation), ' ', inText.lower()))]
    else :
        stemmed = [wrdEdit(i) for i in nltk.tokenize.word_tokenize(inText.lower())]
    rmStop = [i for i in stemmed if i not in stopwords]
    return ' '.join(rmStop)

def preProcessData(df, stmLem = 0, keePuncs = 0):
    # stmLem = 1 for stemming
    # stmLem = 2 for lemmatization
    # stmLem = 0 for none ,i.e, only stopword removal

    dfDict = {'Sentiment':[],'CoronaTweet':[]}
    for i in range(len(df['CoronaTweet'])):
        dfDict['Sentiment'].append(df['Sentiment'][i])
        dfDict['CoronaTweet'].append(processInput(df['CoronaTweet'][i],wordEdit(stmLem),set(nltk.corpus.stopwords.words('english')), keePuncs))
    return dfDict

#Stemming
trainSourceStem = pd.DataFrame(preProcessData(pd.read_csv('data/Corona_train.csv'),1,1))
testSourceStem = pd.DataFrame(preProcessData(pd.read_csv("data/Corona_validation.csv"),1,1))
mainCountStemUni, denomStemUni = getCounts(trainSourceStem,cMap)

matches = 0
for i,j in zip(trainSourceStem["Sentiment"],trainSourceStem["CoronaTweet"]):
    if i == predict(j,[(1,mainCountStemUni,denomStemUni)],cProbs,icMap)[0] :
        matches+=1
trainAccNB = matches/len(trainSourceStem)
print(f'Accuracy over training data : {matches/len(trainSourceStem)}')

matches = 0
modelValPreds = [predict(j,[(1,mainCountStemUni,denomStemUni)],cProbs,icMap)[0] for j in testSourceStem['CoronaTweet']]
for i,j in zip(testSourceStem["Sentiment"],modelValPreds):
    if i == j :
        matches+=1
valAccNB = matches/len(testSourceStem)
print(f'Accuracy over validation data : {matches/len(testSourceStem)}')

# drawWrdCld(mainCountStemUni[0],'plots/partD_positive_wrdCloudStem.png')
# drawWrdCld(mainCountStemUni[1],'plots/partD_negative_wrdCloudStem.png')
# drawWrdCld(mainCountStemUni[2],'plots/partD_neutral_wrdCloudStem.png')

#Lemmatizing
trainSourceLemm = pd.DataFrame(preProcessData(pd.read_csv('data/Corona_train.csv'),2,1))
testSourceLemm = pd.DataFrame(preProcessData(pd.read_csv("data/Corona_validation.csv"),2,1))
mainCountLemmUni, denomLemmUni = getCounts(trainSourceLemm,cMap)

matches = 0
for i,j in zip(trainSourceLemm["Sentiment"],trainSourceLemm["CoronaTweet"]):
    if i == predict(j,[(1,mainCountLemmUni,denomLemmUni)],cProbs,icMap)[0] :
        matches+=1
trainAccNB = matches/len(trainSourceLemm)
print(f'Accuracy over training data : {matches/len(trainSourceLemm)}')

matches = 0
modelValPredsLemm = [predict(j,[(1,mainCountLemmUni,denomLemmUni)],cProbs,icMap)[0] for j in testSourceLemm['CoronaTweet']]
for i,j in zip(testSourceLemm["Sentiment"],modelValPredsLemm):
    if i == j :
        matches+=1
valAccNB = matches/len(testSourceLemm)
print(f'Accuracy over validation data : {matches/len(testSourceLemm)}')

# drawWrdCld(mainCountLemmUni[0],'plots/partD_positive_wrdCloudLemm.png')
# drawWrdCld(mainCountLemmUni[1],'plots/partD_negative_wrdCloudLemm.png')
# drawWrdCld(mainCountLemmUni[2],'plots/partD_neutral_wrdCloudLemm.png')

#Nothing
trainSourceNoth = pd.DataFrame(preProcessData(pd.read_csv('data/Corona_train.csv'), 0))
testSourceNoth = pd.DataFrame(preProcessData(pd.read_csv("data/Corona_validation.csv"),0))
mainCountNothUni, denomNothUni = getCounts(trainSourceNoth,cMap)

matches = 0
for i,j in zip(trainSourceNoth["Sentiment"],trainSourceNoth["CoronaTweet"]):
    if i == predict(j,[(1,mainCountNothUni,denomNothUni)],cProbs,icMap)[0] :
        matches+=1
trainAccNB = matches/len(trainSourceNoth)
print(f'Accuracy over training data : {matches/len(trainSourceNoth)}')

matches = 0
modelValPredsNoth = [predict(j,[(1,mainCountNothUni,denomNothUni)],cProbs,icMap)[0] for j in testSourceNoth['CoronaTweet']]
for i,j in zip(testSourceNoth["Sentiment"],modelValPredsNoth):
    if i == j :
        matches+=1
valAccNB = matches/len(testSourceNoth)
print(f'Accuracy over validation data : {matches/len(testSourceNoth)}')

# drawWrdCld(mainCountNothUni[0],'plots/partD_positive_wrdCloudNoth.png')
# drawWrdCld(mainCountNothUni[1],'plots/partD_negative_wrdCloudNoth.png')
# drawWrdCld(mainCountNothUni[2],'plots/partD_neutral_wrdCloudNoth.png')

########
#PART E#
########

trainSourceStem = pd.DataFrame(preProcessData(pd.read_csv('data/Corona_train.csv'),1,1))
testSourceStem = pd.DataFrame(preProcessData(pd.read_csv("data/Corona_validation.csv"),1,1))

cProbs, cMap, icMap = getClassProbs(trainSourceStem["Sentiment"])
mainCountStemBi, denomStemBi = getCounts(trainSourceStem,cMap,n = 2)
bigramsStemPred = [(1,mainCountStemUni,denomStemUni),(2,mainCountStemBi,denomStemBi)]
print('\nSimple Bi_gram\n')
matches = 0
trainPredStemBiGram = [predict(j,bigramsStemPred, cProbs, icMap)[0] for j in trainSourceStem["CoronaTweet"] ]
for i,j in zip(trainSourceStem["Sentiment"],trainPredStemBiGram):
    if i == j :
        matches+=1
partE_biTrain = matches/len(trainSourceStem)
print(f'Accuracy over training data : {matches/len(trainSourceStem)}')

matches = 0
testPredStemBiGram = [predict(j,bigramsStemPred, cProbs, icMap)[0] for j in testSourceStem["CoronaTweet"] ]
for i,j in zip(testSourceLemm["Sentiment"],testPredStemBiGram):
    if i == j :
        matches+=1
partE_biTest = matches/len(testSourceStem)
print(f'Accuracy over validation data : {matches/len(testSourceStem)}')

print('\nTri-gram with Stemming\n')
mainCountStemTri, denomStemTri = getCounts(trainSourceStem,cMap,n = 3)
trigramsStemPred = [(1,mainCountStemUni,denomStemUni),(2,mainCountStemBi,denomStemBi),(3,mainCountStemTri,denomStemTri)]
matches = 0
trainPredStemTriGram = [predict(j,trigramsStemPred, cProbs, icMap)[0] for j in trainSourceStem["CoronaTweet"] ]
for i,j in zip(trainSourceStem["Sentiment"],trainPredStemTriGram):
    if i == j :
        matches+=1
partE_TriTrain = matches/len(trainSourceStem)
print(f'Accuracy over training data : {matches/len(trainSourceStem)}')

matches = 0
testPredStemTriGram = [predict(j,trigramsStemPred, cProbs, icMap)[0] for j in testSourceStem["CoronaTweet"] ]
for i,j in zip(testSourceLemm["Sentiment"],testPredStemTriGram):
    if i == j :
        matches+=1
partE_TriTest = matches/len(testSourceStem)
print(f'Accuracy over validation data : {matches/len(testSourceStem)}')

print('\nBiGram with Stemming and Punctuation Removal\n')
trainSourceStemPunc = pd.DataFrame(preProcessData(pd.read_csv('data/Corona_train.csv'),1))
testSourceStemPunc = pd.DataFrame(preProcessData(pd.read_csv("data/Corona_validation.csv"),1))

cProbs, cMap, icMap = getClassProbs(trainSourceStemPunc["Sentiment"])
mainCountStemBiPunc, denomStemBiPunc = getCounts(trainSourceStemPunc,cMap,n = 2)
mainCountStemUniPunc, denomStemUniPunc = getCounts(trainSourceStemPunc,cMap,n = 1)
bigramsStemPredPunc = [(1,mainCountStemUniPunc,denomStemUniPunc),(2,mainCountStemBiPunc,denomStemBiPunc)]

matches = 0
trainPredStemBiGramPunc = [predict(j,bigramsStemPredPunc, cProbs, icMap)[0] for j in trainSourceStemPunc["CoronaTweet"] ]
for i,j in zip(trainSourceStemPunc["Sentiment"],trainPredStemBiGramPunc):
    if i == j :
        matches+=1
partE_puncBiTrain = matches/len(trainSourceStemPunc)
print(f'Accuracy over training data : {matches/len(trainSourceStemPunc)}')

matches = 0
testPredStemBiGramPunc = [predict(j,bigramsStemPred, cProbs, icMap)[0] for j in testSourceStemPunc["CoronaTweet"] ]
for i,j in zip(testSourceStemPunc["Sentiment"],testPredStemBiGramPunc):
    if i == j :
        matches+=1
partE_puncBiTest = matches/len(testSourceStemPunc)
print(f'Accuracy over validation data : {matches/len(testSourceStemPunc)}')

#TF-IDF#
trainX = trainSourceNoth['CoronaTweet']
trainY = trainSourceNoth['Sentiment']
testX = testSourceNoth['CoronaTweet']
testY = testSourceNoth['Sentiment']

tf_vectorizer = CountVectorizer() 
trainXtf = tf_vectorizer.fit_transform(trainX)
testXtf = tf_vectorizer.transform(testX)
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(trainXtf, trainY)
predY = naive_bayes_classifier.predict(testXtf)
partE_tfidAcc = np.where((predY == np.array(testY)) == True)[0].shape[0]/predY.shape[0]

########
#PART F#
########

#merging two datasets
def domainMerge(source : pd.DataFrame, domain: pd.DataFrame):
    dfFinal = {"Sentiment" : [], "CoronaTweet" : []}
    for i,j in zip(source['Sentiment'], source['CoronaTweet']):
        dfFinal["Sentiment"].append(i)
        dfFinal["CoronaTweet"].append(j)

    for i,j in zip(domain['Sentiment'], domain['Tweet']):
        dfFinal["Sentiment"].append(i)
        dfFinal["CoronaTweet"].append(j)
    return pd.DataFrame(dfFinal)

domainVal = pd.read_csv('data/Domain_Adaptation/Twitter_validation.csv')
domainVal.rename(columns={'Tweet':'CoronaTweet'},inplace=True)
domainValidation = pd.DataFrame(preProcessData(domainVal,1,1))

def domainAdaptationAccuracies(n):
    print(f'Percent size = {n}')
    domainAdaptation1 = pd.read_csv("data/Domain_Adaptation/Twitter_train_"+str(n)+".csv")
    trainDomain1Clean = pd.DataFrame(preProcessData(domainMerge(trainSourceStem,domainAdaptation1),1,1))
    cProbs, cMap, icMap = getClassProbs(trainDomain1Clean["Sentiment"])
    mainCountDomainUni10, denomDomainUni10 = getCounts(trainDomain1Clean,cMap,n = 1)
    matches = 0
    tot = 0
    trainPredCleanDomain1= [predict(j,[(1,mainCountDomainUni10,denomDomainUni10)],cProbs,icMap)[0] for j in trainDomain1Clean["CoronaTweet"] ]
    for i,j in zip(trainDomain1Clean["Sentiment"],trainPredCleanDomain1):
        tot+=1
        if i == j :
            matches+=1
    trainAccNB = matches/len(trainDomain1Clean)
    print(f'Accuracy over training data : {matches/len(trainDomain1Clean)}')

    matches = 0
    tot = 0
    testPredCleanBiGram = [predict(j,[(1,mainCountDomainUni10,denomDomainUni10)],cProbs,icMap)[0] for j in domainValidation["CoronaTweet"] ]
    for i,j in zip(domainValidation["Sentiment"],testPredCleanBiGram):
        tot+=1
        if i == j :
            matches+=1
    trainAccNB = matches/len(domainValidation)
    print(f'Accuracy over validation data : {matches/len(domainValidation)}')

print("With Source\n")
domainAdaptationAccuracies(1)
domainAdaptationAccuracies(2)
domainAdaptationAccuracies(5)
domainAdaptationAccuracies(10)
domainAdaptationAccuracies(25)
domainAdaptationAccuracies(50)
domainAdaptationAccuracies(100)

def domainAdaptationAccuraciesTarget(n):
    print(f'Percent size = {n}')
    domainAdaptation1 = pd.read_csv("data/Domain_Adaptation/Twitter_train_"+str(n)+".csv")
    domainAdaptation1 = domainAdaptation1.rename(columns={'Tweet':"CoronaTweet"})
    trainDomain1Clean = pd.DataFrame(preProcessData(domainAdaptation1,1))
    cProbs, cMap, icMap = getClassProbs(trainDomain1Clean["Sentiment"])
    mainCountDomainUni10, denomDomainUni10 = getCounts(trainDomain1Clean,cMap,n = 1)
    matches = 0
    tot = 0
    trainPredCleanDomain1= [predict(j,[(1,mainCountDomainUni10,denomDomainUni10)],cProbs,icMap)[0] for j in trainDomain1Clean["CoronaTweet"] ]
    for i,j in zip(trainDomain1Clean["Sentiment"],trainPredCleanDomain1):
        tot+=1
        if i == j :
            matches+=1
    trainAccNB = matches/len(trainDomain1Clean)
    print(f'Accuracy over training data : {matches/len(trainDomain1Clean)}')

    matches = 0
    tot = 0
    testPredCleanBiGram = [predict(j,[(1,mainCountDomainUni10,denomDomainUni10)],cProbs,icMap)[0] for j in domainValidation["CoronaTweet"] ]
    for i,j in zip(domainValidation["Sentiment"],testPredCleanBiGram):
        tot+=1
        if i == j :
            matches+=1
    trainAccNB = matches/len(domainValidation)
    print(f'Accuracy over validation data : {matches/len(domainValidation)}')

print("Withput Source\n")
domainAdaptationAccuraciesTarget(1)
domainAdaptationAccuraciesTarget(2)
domainAdaptationAccuraciesTarget(5)
domainAdaptationAccuraciesTarget(10)
domainAdaptationAccuraciesTarget(25)
domainAdaptationAccuraciesTarget(50)
domainAdaptationAccuraciesTarget(100)

valiAccS = [45.46056991385023,45.46056991385023,46.4546056991385,47.6805831676607,47.514910536779326,48.641484426772696,49.00596421471173]
valiAccNS = [35.48707753479125,39.595758780649437,44.30086149768058,48.60834990059642,48.906560636182905,51.55732273028496,56.03048376408217]
X = [1,2,5,10,25,50,100]
plt.plot(X,valiAccNS, label = 'Accuracy without Source')
plt.plot(X,valiAccS, label = 'Accuracy with Source')
plt.legend()
plt.savefig('plots/partFaccuracies.png')
plt.show()
