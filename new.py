import xlrd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import re
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_file
import nltk
import gensim
import string
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import euclidean_distances
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict
import math
from scipy.spatial import distance
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Open the workbook
xl_workbook = xlrd.open_workbook("VR_AR_DATASET_USPTO.xlsx") 

# Grabbing the Required Sheet by index 
xl_sheet = xl_workbook.sheet_by_index(0)

#Initialising the Lists Required


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

lemmatiser = WordNetLemmatizer()


output_file("test.html")

stemmer = nltk.SnowballStemmer("english")

topics = []
currTopicList = []
overallTopicList = []
input_data = []

def tokenize(text):
    lemmatized_words = []
    text = text.lower()
    text = text.strip("\n")
    text = re.sub(r'\\n',' ',text)
    text = re.sub(r'[^\w\s]',' ',text)
    tokens = nltk.word_tokenize(text)
    tokens_pos = pos_tag(tokens)
    count = 0
    for token in tokens:
        pos = tokens_pos[count]
        pos = get_wordnet_pos(pos[1])
        if pos != '':
            lemma = lemmatiser.lemmatize(token, pos)
        else:
            lemma = lemmatiser.lemmatize(token)
        lemmatized_words.append(lemma)
        count+=1
    return lemmatized_words


num_cols = xl_sheet.ncols   #Number of columns
for row_idx in range(2, xl_sheet.nrows): 
#for row_idx in range(2, 10):   # Iterate through rows

    # for col_idx in range(1, 2):  
    #     cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
    #     var1 = str(cell_obj)[7:-1]
 
    col_idx = 1
    doc1 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 2
    doc2 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 3
    doc3 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 4
    doc4 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx == 5
    doc5 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx == 6
    doc6 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 20
    doc7 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 38
    doc8 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx =65
    doc9 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 69
    doc10 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx =72
    doc11 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx =74
    doc12 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 78
    doc13 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx =79
    doc14 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx =81
    doc15 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    col_idx = 82
    doc16 = str(xl_sheet.cell(row_idx, col_idx))[7:-1]
    doc1 = doc1 + doc2 + doc3 + doc4 + doc5 + doc6 + doc7 + doc8 + doc9 + doc10+ doc11 + doc12 + doc13 + doc14 + doc15 + doc16

    doc_complete = [doc1]


    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]       


    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=1, id2word = dictionary, passes=10)

    #Results
    #curr_topic= str(ldamodel.print_topics(num_topics=3, num_words=3)[0])[12:-2]
    curr_topic= str(ldamodel.print_topics(num_topics=1, num_words=10)[0])

    #print(curr_topic)


    
    currTopicList =re.findall(r'"([^"]*)"', curr_topic)
    #print currTopicList
    abc = currTopicList[0]+' '+currTopicList[1]+' '+currTopicList[2]+' '+currTopicList[3]+' '+currTopicList[4]+' '+currTopicList[5]+' '+currTopicList[6]+' '+currTopicList[7]+' '+currTopicList[8]+' '+currTopicList[9]
    abc = abc.strip("\n")
    input_data.append(abc)

    # N = len(currTopicList)


    # for i in range(0, N): 
    #     overallTopicList.append(currTopicList[i])

    # topics.append(currTopicList)

count_vectorizer = CountVectorizer(encoding="latin-1", stop_words="english", tokenizer=tokenize, analyzer='word')
Tfidf_vectorizer = TfidfVectorizer(encoding="latin-1", use_idf=True, stop_words="english", tokenizer=tokenize, analyzer='word')
vectorizer = TfidfVectorizer(encoding="latin-1", use_idf=True, stop_words="english", tokenizer=tokenize, analyzer='word')
input_freq = Tfidf_vectorizer.fit_transform(input_data).toarray()
input_vector = Tfidf_vectorizer.fit_transform(input_data)

#num_clusters = 5
num_clusters = input("Number of clusters you want: ")
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(input_vector)
clusters = clustering_model.fit_predict(input_vector)

count=1
cluster_text = defaultdict(list)
docs_list =[]

# for cluster in clusters:
#     print cluster
#print clusters[0]
# print clusters


# x = clusters
# y = np.bincount(x)
# ii = np.nonzero(y)[0]
overallclusterlist=[]

count_data = len(input_data)

for i in range(count_data):
    cluster_id = clusters[i]
    doc_id = i+1
    cluster_text[cluster_id].append(doc_id)

cluster_terms = defaultdict(list)
cluster_top_terms=[]
# print("Top terms per cluster:")
centroids = clustering_model.cluster_centers_
order_centroids = centroids.argsort()[:, ::-1]
terms = Tfidf_vectorizer.get_feature_names()
# for i in range(num_clusters):
#     #cluster_top_terms[i]=''
#     abc = ''
#     print "Cluster %d:" % i,
#     for ind in order_centroids[i, :50]:
#         abc= abc + ' '+ terms[ind]
#         #print terms[ind]
#          #print ' %s' % terms[ind],
#     cluster_top_terms[i] = abc
# print cluster_top_terms
#         cluster_terms[i].append((ind,terms[ind]))

#     print
# print cluster_terms
# Add a bold format to use to highlight cells.


for i in range(num_clusters):
    cluster_list=[]

    for j in range ( len(input_data)):
        if clusters[j]== i :
            #print "hi"
            cluster_list.append(j)

    overallclusterlist.append(cluster_list)

import xlsxwriter

workbook = xlsxwriter.Workbook('result.xlsx')
worksheet = workbook.add_worksheet()
bold = workbook.add_format({'bold': True})
documentnum= []
count = num_clusters
row = 0
k = 0

while (row<count ):
    worksheet.write(k, 0, "Cluster Number",       bold)
    worksheet.write(k, 1, row)
    worksheet.write(k, 2, "Top Words",       bold)

    abc = ''
    for ind in order_centroids[row, :50]:
        abc= abc + ' '+ terms[ind]
    worksheet.write(k, 3, abc)



    k = k+1
    worksheet.write(k, 0, "Serial No",       bold)
    worksheet.write(k, 1, "Patent ID",       bold)
    worksheet.write(k, 2, "Patent Title",       bold)
    worksheet.write(k, 3, "Top Terms",       bold)
    k = k +1

    documentnum = []

    # count2 = len(overallclusterlist[row])
    # j =0 
    # while ( j< count2):
    #     N1 = len(topics[j])
    #     i = 0
    #     while ( i < N1 ):
    #         if topics[j][i] == sorted_x[row][0]:

    #             documentnum.append(j+1)

    #         i = i+1
    #     j = j+1

    # this first kind of for-loop goes through a list
    i = 0
    for number in overallclusterlist[row]:
        row_idx = number + 2
        i = i +1 

        worksheet.write(k, 0, i)

        for col_idx in range(0, 1):  # Iterate through article_citation column
            cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
            citation_count = str(cell_obj)[7:-2]
            worksheet.write(k, 1, citation_count)
        for col_idx in range(1, 2):  # Iterate through article_citation column
            cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
            citation_count = str(cell_obj)[7:-2]
            worksheet.write(k, 2, citation_count)

        worksheet.write(k, 3, input_data[number])

        k =k + 1   
    # sum2 = sum2 + sorted_x[row][1] + 1
    

    row = row +1
workbook.close() 

    



# dictTopicList = {x:overallTopicList.count(x) for x in overallTopicList}
# import operator
# sorted_x = sorted(dictTopicList.items(), key=operator.itemgetter(1),reverse=True)
# print sorted_x

# import xlsxwriter

# workbook = xlsxwriter.Workbook('sample2.xlsx')
# worksheet = workbook.add_worksheet()
# documentnum= []
# count = len (sorted_x)
# row = 0
# k = 0
# sum2 = 0
# while (row<count ):
#     worksheet.write(k, 0, "Classification")
#     worksheet.write(k, 1, sorted_x[row][0])
#     worksheet.write(k, 2, "Count")
#     worksheet.write(k, 3, sorted_x[row][1])
#     k = k+1
#     worksheet.write(k, 0, "Paper No")
#     worksheet.write(k, 1, "Article Citation Count")
#     worksheet.write(k, 2, "Document Title")
#     worksheet.write(k, 3, "Year")
#     k = k +1

#     documentnum = []

#     count2 = len(topics)
#     j =0 
#     while ( j< count2):
#         N1 = len(topics[j])
#         i = 0
#         while ( i < N1 ):
#             if topics[j][i] == sorted_x[row][0]:

#                 documentnum.append(j+1)

#             i = i+1
#         j = j+1

#     # this first kind of for-loop goes through a list
#     for number in documentnum:
#         row_idx = number + 1

#         for col_idx in range(0, 1):  # Iterate through article_citation column
#             cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
#             citation_count = str(cell_obj)[7:-2]
#             worksheet.write(k, 0, citation_count)

#         for col_idx in range(21, 22):  # Iterate through article_citation column
#             cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
#             citation_count = str(cell_obj)[7:-2]
#             if citation_count == ' ' :
#                 citation_count = 0
#             if citation_count == '' :
#                 citation_count = 0
#             worksheet.write(k, 1, citation_count)

#         for col_idx in range(1, 2):  # Iterate through article_citation column
#             cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
#             citation_count = str(cell_obj)[7:-1]
#             worksheet.write(k, 2, citation_count)

#         for col_idx in range(6, 7):  # Iterate through article_citation column
#             cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
#             citation_count = str(cell_obj)[7:-2]
#             worksheet.write(k, 3, citation_count)

#         k =k + 1   
#     # sum2 = sum2 + sorted_x[row][1] + 1
    

#     row = row +1
# workbook.close() 



# writer = pd.ExcelWriter('Classification_count.xlsx', engine='openpyxl')

# row = 0
# k = 0
# count = len (sorted_x)

# xl_workbook = xlrd.open_workbook("sample2.xlsx") 
# xl_sheet = xl_workbook.sheet_by_index(0)
# num_rows = xl_sheet.nrows

# while (row<count):
#     z= num_rows - (k + sorted_x[row][1] + 2)
#     y= num_rows - k 
#     df2= pd.read_excel('sample2.xlsx', sheetname='Sheet1' , skiprows = k, skip_footer= y)
#     df = pd.read_excel('sample2.xlsx', sheetname='Sheet1' , skiprows = k+1, skip_footer= z)
#     df1 = df.sort(['Article Citation Count', 'Year'], ascending=[False, False])
#     df2.to_excel(writer,'Sheet1',index=False, startrow = k)
#     df1.to_excel(writer,'Sheet1',index=False, startrow = k+1)
#     writer.save()
#     k = k +sorted_x[row][1] + 2
#     row = row +1



# import os
# os.remove("sample2.xlsx")


