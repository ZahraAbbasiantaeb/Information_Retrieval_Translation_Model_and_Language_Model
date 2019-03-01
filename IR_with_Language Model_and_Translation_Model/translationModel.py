import pickle
import nltk
from gensim.models import Word2Vec
import hickle as hkl
from readCorpus import evaluation, getGold, data_path, TF_path, writeToFile, translation_results_path, \
    word2vec_corpus_path, readFile


# we need to load modeled TF array and other needed data which were saved with pickle and hickle
with open(data_path, "rb") as f:
    document_id, distinct_words, docs_size, CF, QID, title, description, judge, titleAndDescription= pickle.load(f)

TF_array= hkl.load(TF_path)


#this func creates word2VecModel of given corpus path
def createword2VecModel(corpus):

    model = Word2Vec(corpus,window=5, min_count=1, workers=4)

    model.save("word2vec_Hamshahri.model")

    return



# this function removes unwanted punctuations from given text
def processText(text):

    sentences=[]

    res=text.split(".")

    for sentence in res:

        sentence = sentence.replace('،',' ')
        sentence = sentence.replace(':', ' ')
        sentence = sentence.replace(';', ' ')
        sentence = sentence.replace('؟', ' ')
        sentence = sentence.replace('!', ' ')
        sentence = sentence.replace('٪', ' ')
        sentence = sentence.replace(')', ' ')
        sentence = sentence.replace('(', ' ')

        sentences.append(nltk.word_tokenize(sentence))

    return sentences



#this function loads the implemented word2vec model
def loadModel():

    model = Word2Vec.load("word2vec_Hamshahri.model")

    return model



#this func evaluates all given queries with given precisions (@k)
def evalqueries(queries, precisions):

    tmp = 0

    f = open(translation_results_path, 'a')

    for query in queries:

        # print(query)

        res = evalOneQuery(nltk.word_tokenize(query))

        # print(res[:20])

        for precision in precisions:

            precisionAtK = evaluation(res, getGold(QID[tmp],judge), precision)

            template="precision @ %s  for QID %s  is: %s \n"

            f.write("\n")

            # print(template %(precision ,QID[tmp] , precisionAtK))

            f.write(template %(precision ,QID[tmp] , precisionAtK))

            writeToFile(f,res[:20])

        tmp += 1
    return




# this func chooses a subsample of TF array for given query
def createTFForQuery(query_terms):

    index = 0

    TempTF = []

    words = []

    for word in distinct_words:

        if word in query_terms:

            TempTF.append(TF_array[index,:])

            words.append(word)

        index += 1

    return TempTF,words



def getVocabList(vocabs):

    res=[]

    for vocab in vocabs:

        for item in vocab:

            res.append(item[0])

    return res



# this func is used to evaluate one query
def evalOneQuery(query_terms):

    query_words = []

    vocabs = []

    for word in query_terms:

        vocab, sum = getModel(word)
        query_words.append((vocab,sum))
        vocabs.append(vocab)

    words = getVocabList(vocabs)

    TF_sample, distinct_sample = createTFForQuery(words)

    index = 0

    results = []

    for doc in document_id:

        similarity = sim_of_query_to_doc(index,query_words,TF_sample,distinct_sample)

        results.append((doc,similarity))

        index+= 1

    return sorted(results, key=lambda t: t[1], reverse=True)



# this function returns the first 20 most similar words to the given word
def getModel(word):

    simWords = []

    if word in model:

        simWords=model.most_similar(word, topn=20)

    sum = 0

    for set in simWords:

        sum += set[1]

    simWords.append((word, sum))

    return simWords, sum



#this function returns the relevance degree of given doc to given query
def sim_of_query_to_doc(docID,query_words,TF_sample,distinct_sample):

    res = []
    similarity = 1

    for term in query_words:
        vocab = term[0]
        sum = term[1]

        #sigma for expansion of one query word

        for item in vocab:
            sigma = 0.00000000000000000000001

            if(item[0] in distinct_sample):

                index = distinct_sample.index(item[0])

                sigma += (TF_sample[index][docID]/docs_size[docID])*(item[1]/sum)

        res.append(sigma)

    for num in res:

        similarity *= num

    return similarity



model = loadModel()

k_array=[5, 10, 20]

evalqueries(title, k_array)

print('******')

evalqueries(description, k_array)

print('******')

evalqueries(titleAndDescription, k_array)

createword2VecModel(processText(readFile(word2vec_corpus_path)))
