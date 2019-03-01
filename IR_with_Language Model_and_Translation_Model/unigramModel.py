import numpy as np
import pickle
from nltk import word_tokenize
import hickle as hkl
from readCorpus import data_path, results_path, getGold, writeToFile, evaluation, TF_path


u_coefficient=1

# we need to load modeled TF array and other needed data which were saved with pickle and hickle
with open(data_path, "rb") as f:
    document_id, distinct_words, docs_size, CF, QID, title, description, judge, titleAndDescription= pickle.load(f)

TF_array= hkl.load(TF_path)

corpus_size=np.sum(docs_size)


# this function evaluates set of given queries with given parameters which are used as k in precision@k
# and writes results to 'results_path' file
def evalqueries(queries, precisions):

    query_ID = 0

    f = open(results_path,'a')

    for query in queries:

        res = evalOneQuery(word_tokenize(query))

        # print(res[:20])

        for precision in precisions:

            precisionAtK = evaluation(res, getGold(QID[query_ID],judge), precision)

            template="precision @ %s  for QID %s  is: %s \n"

            f.write("\n")

            print (template %(precision ,QID[query_ID] , precisionAtK))

            f.write(template %(precision ,QID[query_ID] , precisionAtK))

            writeToFile(f,res[:20])

        query_ID += 1

    return



# this func chooses a subsample of TF array for given query
def createTFForQuery(query_terms):

    index=0

    TempTF=[]

    for word in distinct_words:

        if word in query_terms:

            TempTF.append(TF_array[index,:])

        index += 1

    return TempTF



# this function returns the ranked documents according to given query
def evalOneQuery(query_terms):

    index = 0

    results = []

    tempTF = createTFForQuery(query_terms)

    for doc in document_id:

        similarity =sim_of_query_to_doc(index, tempTF)

        results.append((doc, similarity))

        index+= 1

    return sorted(results, key=lambda t: t[1], reverse=True)



# this function calculates similarity of one query with given document
def sim_of_query_to_doc(docID, tempTF):

    similarity = 1

    tmp = 0

    row = np.shape(tempTF)[0]

    for i in range (0,row):

        if (CF[i]>0):

            tmp = 1

            arg = (tempTF[i][docID]+((u_coefficient*CF[i])/corpus_size))/(docs_size[docID]+u_coefficient)

            similarity *= arg

    if(tmp == 0):

        similarity = 0

    return similarity


k_array=[5, 10, 20]
evalqueries(title, k_array)
evalqueries(description, k_array)
evalqueries(titleAndDescription, k_array)