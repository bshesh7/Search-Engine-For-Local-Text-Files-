import os
import math
import operator
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from collections import OrderedDict

stemmer = PorterStemmer()
documents ={}
stop_word = stopwords.words('english')

#these are for calculating the idf
aldocuments = {}
temp_list = []
corpusroot = './athletes'
dict_postinglist = {}
file_list = []

for filename in os.listdir(corpusroot):
    file_list.append(filename)
    listOfNoDuplicates = []
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    for index, word in enumerate(tokens):
        if word not in stop_word:
            wd = stemmer.stem(word)
            listOfNoDuplicates.append(wd)
            temp_list.append(wd)

    t = dict(Counter(listOfNoDuplicates)) #this counts the numbers os words in that particular file and stores in a dictionary "t"
    documents[filename]=t #that dictionary in each forloop belongs to a specific file so we create a new dictionary with key of filename and value of t dictionary which is list of words and its counts

aldocuments = temp_list
#dict_postinglist = file_list
#print(dict_postinglist)
#print(documents['1960-09-26.txt']) just for checking

#getidf function
def getidf(tok):
    count = 0
    for sentence in documents:
        if tok in documents[sentence]:
            count += 1
    if count == 0:
        return -1
    idf = math.log(len(documents)/count,10)
    return idf
raw_weights_dict = {}
normaliser_dict = {}
norm_weights_dict = {}

def weight_calculation():
    #calculating raw idf
    for filename in documents:
        normalizer = 0
        for tokens in documents[filename]:
            raw_weights_dict_t = {}
            raw_tfidf = (1 + math.log(documents[filename][tokens],10)) * getidf(tokens)
            documents[filename][tokens] = raw_tfidf
            # also calculate the diviser for normalising
            normalizer = normalizer + (raw_tfidf * raw_tfidf)
        normalizer = math.sqrt(normalizer)
        normaliser_dict[filename] = normalizer
    #normalising idf
    for filename in documents:
        for tokens in documents[filename]:
            documents[filename][tokens] = documents[filename][tokens] / normaliser_dict[filename]

def getweight(file, tok):
    if tok not in documents[file]:
        return 0
    return documents[file][tok]

def query(senten):
    ag_doc_idf = {}
    doc_similarity = {}
    senten = senten.lower()
    tokens = tokenizer.tokenize(senten)
    NoDuplicates = []
    for index, word in enumerate(tokens):
        if word not in stop_word:
            wd = stemmer.stem(word)
            NoDuplicates.append(wd)
    t = dict(Counter(NoDuplicates))
    query_vec = t
    #print(query_vec)
    #calculate the term frequeny
    normalizer = 0
    for tokens in query_vec:
        raw_tf = (1 + math.log(query_vec[tokens], 10))
        query_vec[tokens] = raw_tf
        normalizer = normalizer + (raw_tf * raw_tf)
    normalizer = math.sqrt(normalizer)
    count2 = 0
    #calculates the weight of tokens in query

    for tokens in query_vec:
        query_vec[tokens] = query_vec[tokens]/normalizer
        count2 = count2 + 1
    count2 = 0
    #until here we find the tfidf weight of the query

    for tokens in query_vec:
        dict_postinglist[tokens] = {}
    #print(dict_postinglist)

    postinglist = {}
    for tokens in query_vec:
        postinglist[tokens] = {}
    # print(dict_postinglist)

    for tokens in query_vec:
        lisst = []
        for filename in documents:
            dict_postinglist[tokens][filename] = getweight(filename, tokens)
            lisst.append(getweight(filename, tokens))
        lisst.sort(reverse=True)
        #print(lisst)
        ct = 0
        for elements in lisst:
            if ct == 9:
                break
            for filename in documents:
                if dict_postinglist[tokens][filename] == elements:
                    postinglist[tokens][filename] = elements
                    #print(elements)
                    #print(filename)
            ct = ct + 1
        #print(postinglist)
    #find unique documents
    unique_doc =[]
    for tokens in postinglist:
        for file in postinglist[tokens]:
            if file not in unique_doc:
                unique_doc.append(file)
    #print(unique_doc)
    fetchmorelist = []
    ctr = 0
    final_dict = {}
    final_dict_sort= {}
    for elements in unique_doc:
        for tokens in postinglist:
                if elements in postinglist[tokens]:
                    ctr = ctr + 1
        count2 = 0
        numerator = 0
        count3 = 0
        for tok in query_vec:
            ag_doc_idf[count2] = getweight(elements, tok)
            count2 = count2 + 1

        if ctr == len(postinglist):

            for tok in query_vec:
                numerator = numerator + (query_vec[tok] * ag_doc_idf[count3])
                count3 = count3 + 1
            final_dict[elements] = numerator
        elif ctr < len(postinglist):
            for tokens in query_vec:
                if elements not in postinglist[tokens]:
                    minlist = []
                    for uni in unique_doc:
                        minlist.append(getweight(uni, tokens))
                    numerator = numerator + (query_vec[tokens] * min(minlist))
                    count3 = count3 + 1
                else:
                    numerator = numerator + (query_vec[tokens] * ag_doc_idf[count3])
                    count3 = count3 + 1
            final_dict[elements] = numerator

            fetchmorelist.append(elements)


    #print(final_dict);
    #sort final_dict
    final_dict_sort = sorted(final_dict.items(),key = operator.itemgetter(1),reverse = True)

    """
    if final_dict_sort[0][1] == 0:
        return("None",0.000000000000)
    """
    #print(fetchmorelist)
    #if final_dict_sort[0][0] in fetchmorelist:
       # return ("fetch more", 0.000000000000)
    #if final_dict_sort[0][1] == 0:
     #   return ("None", 0.000000000000)
    return(final_dict_sort[0][0],final_dict_sort[0][1])


#print(getweight("2012-10-16.txt","hispan"))
#print("%.12f" % getweight("2012-10-03.txt","health"))
#print("%.12f" % getweight("1976-10-22.txt","agenda"))
#print("%.12f" % ufgetweight("2012-10-16.txt","hispanic"))
#print("%.12f" % getidf("hispanic"))
weight_calculation()
#query("terror attack")

#test case
#print("%.12f" % getidf("health"))
#print("%.12f" % getidf("agenda"))
#print("%.12f" % getidf("vector"))
#print("%.12f" % getidf("reason"))
#print("%.12f" % getidf("hispan"))
#print("%.12f" % getidf("hispanic"))
#print("%.12f" % getweight("2012-10-03.txt","health"))
#print("%.12f" % getweight("1960-10-21.txt","reason"))
#print("%.12f" % getweight("1976-10-22.txt","agenda"))
#print("%.12f" % getweight("2012-10-16.txt","hispan"))
#print("%.12f" % getweight("2012-10-16.txt","hispanic"))
#print("(%s, %.12f)" % query("health insurance wall street"))

#print("(%s, %.12f)" % query("particular constitutional amendment"))
#print("(%s, %.12f)" % query("terror attack"))
#print("(%s, %.12f)" % query("vector entropy"))
print("(%s, %.12f)" % query("madrid"))


