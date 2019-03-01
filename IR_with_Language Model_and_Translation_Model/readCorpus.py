from nltk.tokenize import word_tokenize
import nltk as nltk
import numpy as np
import pickle
import hickle as hkl

# this part defines path of data files
directory='path_to_data_directory'

corpus_path= directory+'Hamshahri-Corpus.txt'

query_Path= directory+'Querys.txt'

judgment_Path= directory+'judgment.txt'

results_path= directory+'result.txt'

data_path=directory+'dataModel.txt'

TF_path=directory+'TFModel.txt'

translation_results_path=directory+ 'translation_result.txt'

word2vec_corpus_path=directory+'word2vec_train.txt'


def parseXML(path, tag):

    begin_str = '<'+tag+'>'

    end_str = '</'+tag+'>'

    file = open(path,'r')

    begin = 0

    value = []

    docInfo = ""

    for line in file:

        match = line.find(begin_str)

        if match == 0:

            begin = 1

        match2 = line.find(end_str)

        if match2 == 0:

            begin = 0

            docInfo = docInfo.replace("\n","")

            docInfo = prepareString(docInfo)

            value.append(docInfo)

            docInfo = ""

        elif begin == 1 and match == -1 and match2 == -1:

            docInfo += line

    return value


# this func returns the content of a file.
def readFile(path):

    file = open(path,"r")

    text = file.read()

    return text;


# trims a persian string by omitting some punctuations
def prepareString(string):

    string=  string.replace('،', ' ')
    string = string.replace('.', ' ')
    string = string.replace(':', ' ')
    string = string.replace('؛', ' ')
    string = string.replace(')', ' ')
    string = string.replace('(', ' ')
    string = string.replace('!', ' ')
    string = string.replace('?', ' ')
    string = string.replace('<', ' ')
    string = string.replace('>' ,' ')
    string = string.replace('[', ' ')
    string = string.replace(']', ' ')
    string = string.replace('{', ' ')
    string = string.replace('}', ' ')
    string = string.replace('/', ' ')
    string = string.replace('\\', ' ')
    string = string.replace('-', ' ')
    string = string.replace('_', ' ')


    return string


# this func is used to parse the corpus file, it returns array of DID, Corpus, distinct words in all documents, and
# count of words in each document

def parseFile(path):

    file = open(path,'r')

    lineNum = 0

    begin = 0

    docs = []

    docs_info = []

    wordCount_per_doc = []

    doc = ""

    corpus = ""

    for line in file:

        lineNum += 1
        match = line.find('.DID')

        if match == 0:

            str = line.replace('.DID','')
            str = str.replace('\t','')
            str = str.replace('\n','')
            docs_info.append(str.lower())

        if begin == 1:

            has_match = line.find('.DID')

            if has_match > -1:

                begin = 0
                docs.append(prepareString(doc))
                corpus += doc
                wordCount_per_doc.append(len(nltk.word_tokenize(doc)))
                doc = ""

            else:

                doc += line

        elif begin==0:

            has_match = line.find('.Cat')

            if has_match > -1:
                begin = 1;

    docs.append(prepareString(doc))

    corpus += doc

    corpus_tokens = nltk.word_tokenize(corpus)

    distinct_words = set(corpus_tokens)

    wordCount_per_doc.append(len(nltk.word_tokenize(doc)))

    return docs, docs_info, distinct_words, wordCount_per_doc



# this function calculates term frequency per corpus in array called TF_IDF_array
def calculate_TF(TF_array, corpus,distinct_words_in_corpus):

    docID = 0

    for doc in corpus:

        words = word_tokenize(doc)

        for word in words:

            i, = np.where(distinct_words_in_corpus == word)

            TF_array[i, docID] += 1

        docID += 1

    return



# this function calculates IDF for each word in corpus and writes it in array named IDF
def calculate_IDF(CF, TF_array):

    row = np.shape(TF_array)[0]

    for term in range(0, row):

        df = np.sum(TF_array[term, :])

        CF[term, 0] = df

    return



# this func reads judgments from given file
def parseJudgment(path):

    judges = []

    file = open(path,'r')

    for line in file:

        words = nltk.word_tokenize(line)

        judges.append(words)

    return judges



# this func returns the gold data related to given QID
def getGold(QID, judge):

    row = np.shape(judge)[0]

    goldData = []

    for i in range(0,row):

       if QID == judge[i][0]:

           goldData.append(judge[i][1])

    return goldData



# this func writes given results to given file
def writeToFile(f, results):

    f.write("[")

    for ID, precision in results:

        f.write("\n".join(["(%s , %s)" % (ID, precision)]))

    f.write("]\n")

    return



# this func combine two arrays of str
def combineTwoArrays(title, description):

    res=[]

    tmp=len(title)

    for i in range(0,tmp):

        str = title[i] + " "

        str += description[i] + " "

        res.append(str)

    return res



# this func evaluates given results based on gold data with precision@k measure
def evaluation(result, goldData, k):

    tmp = 0

    for i in range(0, k):

        if result[i][0] in goldData:
            tmp += 1

    return (tmp / k)


# read corpus file and save it
def loadAndSaveNeededData():

    corpus, document_id, distinct_words_in_corpus, corpus_size = parseFile(corpus_path)

    distinct_words_in_corpus = list(distinct_words_in_corpus)
    distinct_words_in_corpus = np.transpose(distinct_words_in_corpus)
    CF = np.zeros([np.shape(distinct_words_in_corpus)[0], 1])

    # create a TF-IDF array and fill it using above functions
    TF_array = np.zeros([np.shape(distinct_words_in_corpus)[0], np.shape(document_id)[0]])


    # set TF_IDF array
    calculate_TF(TF_array, corpus, distinct_words_in_corpus)
    calculate_IDF(CF,TF_array)


    # extract data from Query files
    QID = parseXML(query_Path, 'QID')
    title = parseXML(query_Path, 'title')
    description = parseXML(query_Path, 'description')


    # extract data from judgment data file
    judge = parseJudgment(judgment_Path)
    titleAndDescription = combineTwoArrays(title, description)


    with open(data_path,'wb') as f:
        pickle.dump((document_id, distinct_words_in_corpus, corpus_size, CF, QID, title, description, judge, titleAndDescription), f)

    hkl.dump(TF_array, TF_path)

    return


