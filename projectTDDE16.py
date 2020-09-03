import pandas as pd
import spacy
import nltk
import plotly
import plotly.graph_objs as go
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from TDDE16plotly import xyzoffline,plotter       <== Get your own Plotly account. The code for the fucntion is not uploaded. Extraneous.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from bs4 import BeautifulSoup
import os
import re
import csv
import collections
import sys
import random

def row_writer(file_path,ce):
#write cluster counts
    headings = [] 
    headings.extend(ce)
    file_dataset_csv = open(file_path, 'w')
    with file_dataset_csv:
        writer = csv.DictWriter(file_dataset_csv, fieldnames=headings)
        writer.writeheader()
        writer.writerow(clusters3)
    file_dataset_csv.close()


def add_trace(x,y,z):
    r = lambda: random.randint(0,255)
    opacity_rand = lambda: random.randrange(0,255)/(255+10)
    trace = go.Scatter3d(
        x=x[0],    
        y=y[0],      
        z=z[0],    
        mode='markers',
        marker=dict(
            color='#{:02x}{:02x}{:02x}'.format(r(),r(),r()),
            size=3,
            symbol='circle',
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=float('{}'.format(opacity_rand()))
        )
    )
    return trace


def display_word_cloud(embeddings,label,indy=1):

    """ Visulaization function for TSNE cloud """

    x = []
    y = []
    z = []
    for value in embeddings:
        x.append(value[0])
        y.append(value[1])
        z.append(value[2])

    tsne_fig = plt.figure(figsize=(15, 15))  
    tsnex = Axes3D(tsne_fig)
    
    for i in range(len(x)):
        tsnex.scatter(x[i],y[i],z[i])
        #Label the plot! Cool
        #tsnex.text(x[i],y[i],z[i], '%s' % (label[i]) )
    tsnex.set_xlabel('x')
    tsnex.set_ylabel('y')
    tsnex.set_zlabel('z')
    if indy==2:
        #xyzoffline([x],[y],[z],'Word Cloud Doc2Vec')
        plt.savefig(cwd +'\\'+'dataset'+'\\'+'tSNE_word_cloud_{}'.format('d2vec')+'.png')
    #xyzoffline([x],[y],[z],'Word Cloud TF-IDF')
    plt.savefig(cwd +'\\'+'dataset'+'\\'+'tSNE_word_cloud_{}'.format('tf_idf')+'.png')
    #plt.show()


# ----- ---------------------------------------------------------- Main --------------------------------------------------------------------
cwd = os.getcwd()
# There should be a dataset subfolder, if not, create it.
feature_file_path = cwd + "\\" + "dataset"
if not os.path.exists(feature_file_path):
    os.makedirs(feature_file_path)

# The dataset is in European Patents Office DOCDB XML markup. The following patterns are 
# based on the format.
neg_ptrn_1 = re.compile(r'\<exch:patent-classifications\>.*\<\/exch:patent-classifications\>')
neg_ptrn_2 = re.compile(r'\<exch:classifications-ipcr\>.*\<\/exch:classifications-ipcr\>')
neg_ptrn_3 = re.compile(r'\<exch:classifications-ipc\>.*\<\/exch:classifications-ipc\>')
neg_ptrn_4 = re.compile(r'\<exch:dates-of-public-availability\>.*\<\/exch:dates-of-public-availability\>')
neg_ptrn_5 = re.compile(r'\<exch:priority-claims\>.*\<\/exch:priority-claims\>')
neg_ptrn_6 = re.compile(r'\<exch:application-reference[^\s]*data-format\=\"epodoc\"\>.*\<\/exch:application-reference\>') #[\s\w*\-?\w*\=\"\w*\"]*
neg_ptrn_7 = re.compile(r'\<exch:application-reference[\s\w*\-?\w*\=\"\w*\"]*data-format\=\"original\"\>.*\<\/exch:application-reference\>')

#Note: Abstracts: recover those written in english EN only
ptrn_abstract_metadata = re.compile(r'\<exch:abstract[\s\w*\-?\w*\=\"\w*\"]*lang\=\"en\"[\s\w*\-?\w*\=\"\w*\"]*\>.*\<\/exch:abstract\>')
ptrn_title_metadata =  re.compile(r'\<exch:invention-title[\s\w*\-?\w*\=\"\w*\"]*lang\=\"en\"[\s\w*\-?\w*\=\"\w*\"]*\>.*\<\/exch:invention-title\>')

# Patterns to strip Abstract down to content, followed by those for Title
ptrn_abstract_actual_neg1 = re.compile(r'\<exch:abstract\s*.*\<exch:p\>')
ptrn_abstract_actual_neg2 = re.compile(r'\<\/exch:p\>\<\/exch:abstract\>')
ptrn_abstract_actual_neg3 = re.compile(r'PURPOSE:')
ptrn_abstract_actual_neg4 = re.compile(r'CONSTITUTION:.*')

ptrn_title_actual_neg2 = re.compile(r'\<exch:invention-title[\s\w*\-?\w*\=\"\w*\"]*lang\=\"en\"[\s\w*\-?\w*\=\"\w*\"]*\>')
ptrn_title_actual_neg1 = re.compile(r'\<\/exch:invention-title\>')

# Patterns to extract the ONLY needed features: family_id; kind; country etc. Technical details of the features 
# are available in 'How to use' documents published by EPO.
ptrn_family_id = re.compile(r'family-id\=\"\d+\"')
ptrn_kind = re.compile(r'kind\=\"\w+\"')
ptrn_country = re.compile(r'country\=\"\w+\"')
ptrn_doc_number = re.compile(r'doc-number\=\"\w+\"')
ptrn_doc_id = re.compile(r'doc-id\=\"\d+\"')
ptrn_is_rep = re.compile(r'is-representative\=\"[A-Z]+\"')

# if data had been processed prior, skip creating 'dataset.csv' again
file_path = cwd + '\\dataset\\' + 'dataset.csv' 
if not os.path.exists(file_path):
    bag = str()
    data = list()
    line_count = 0
    idx = 0
    # The DOCDB format of patents accepted are published weekly. Grabbed one for use, from which dataset.csv is created
    infile = open(cwd + "\\" + "DOCDB-201731-015-JP-0405.xml","r")
    contents = infile.read()

    # Use lxml XML parser with bs4
    soup = BeautifulSoup(contents,'xml')

    # Write the headers in dataset.csv
    file_dataset_csv = open(file_path, 'a')
    with file_dataset_csv:
        file_dataset_csv.write('Document_Number'+'|'+'Country'+'|'+'Abstract'+'|'+'Family_ID'+'|'+'Title' +'\n')
    file_dataset_csv.close()

    for doc in soup.find_all('exch:exchange-document'):
        idx = idx + 1
        #if idx == 15: #500 for 175
        #    break   
        bag = str(doc)
        # Redactions : metadata not needed
        bag = neg_ptrn_1.sub('',bag)
        bag = neg_ptrn_2.sub('',bag)
        bag = neg_ptrn_3.sub('',bag)
        bag = neg_ptrn_4.sub('',bag)
        bag = neg_ptrn_5.sub('',bag)
        bag = neg_ptrn_6.sub('',bag)
        bag = neg_ptrn_7.sub('',bag)
        
        # Commence Feature Extraction
        find_family_id = re.findall(ptrn_family_id,bag)
        find_family_id = re.findall(r'\d+',find_family_id[0]).pop(0)
        find_family_id = find_family_id.strip()
        
        find_kind = re.findall(ptrn_kind,bag).pop(0)
        find_country = re.findall(ptrn_country,bag).pop(0)
        
        ## Get the actual country code
        find_country = find_country.replace('country="','')
        find_country = find_country.rstrip('"')

        find_doc_number = re.findall(ptrn_doc_number,bag).pop(0)
        ## Clean patent number after extracting it
        find_doc_number = find_doc_number.replace('doc-number="','')
        find_doc_number = find_doc_number.rstrip('"')

        find_doc_id = re.findall(ptrn_doc_id,bag).pop(0)
        find_is_rep = re.findall(ptrn_is_rep,bag).pop(0)
        
        try:
            find_title = re.findall(ptrn_title_metadata,bag).pop(0)
            # clean title
            find_title = ptrn_title_actual_neg1.sub('',find_title)
            find_title = ptrn_title_actual_neg2.sub('',find_title)
            find_title = find_title.lower()
            
        except:
            pass
        
        try:
            find_abs_meta = re.findall(ptrn_abstract_metadata,bag).pop(0)
            # Clean abstract after extracting it
            find_abs_meta = ptrn_abstract_actual_neg1.sub('',find_abs_meta)
            find_abs_meta = ptrn_abstract_actual_neg2.sub('',find_abs_meta)
            find_abs_meta = ptrn_abstract_actual_neg3.sub('',find_abs_meta)
            find_abs_meta = ptrn_abstract_actual_neg4.sub('',find_abs_meta)
            
        except:
            # Some of the documents will not have abstracts, most likely those of kind=U.
            # It means there's a relative classed as kind= A. Again, reflect to EPO DOCDB
            # technical documents to understand the patent terminologies 
            continue

        data.extend([find_doc_number,find_country,find_abs_meta,find_family_id, find_title])
        file_dataset_csv = open(cwd + '\\dataset\\' + 'dataset.csv', 'a')
        with file_dataset_csv:
            file_dataset_csv.write(data[0] + '|' + data[1] + '|' + data[2][:] + '|' + data[3] + '|' + data[4] + '\n')
        line_count = line_count + 1
        data.clear()
    file_dataset_csv.close()
    print("Total Number of Rows Written: ", line_count,'\n')
    print("Prcessing...: Stop words, lemmatization etc.\n")
   
    # ================================ FURTHER DATA PROCESSING =====================================
    #read pre-existing dataset.csv

    #Note: In the previous step of writing to csv, an indexing column is auto written.
    #      This becomes a problem when reading in, since that column is auto-headered as 'Unnamed: 0'.
    #      So it can be dropped. Toggle the commented code line as necessary if using this.
    df = pd.read_csv(cwd+'\\'+'dataset'+'\\'+'dataset.csv',sep='|',error_bad_lines=False, header=0) #.drop(['Unnamed: 0'], axis=1)
    
    # Data-preprocessing: take out stops words [English]
    df['Title'] = df['Title'].str.lower()
    df['Abstract'] = df['Abstract'].str.lower()
    stop_words = set(stopwords.words('english'))
    number_of_rows = len(df.index)

    i = 0
    word_tokens = []
    abstract_as_documents = []
    abstract_as_split_documents = []
    vocab = []
    avgtoks = 0
    while i < number_of_rows:
        cntoks = 0
        # tokenize, clear-out stop-words and punctuations
        for token in word_tokenize(df['Title'][i]):
            cntoks = cntoks + 1  
            if not token in stop_words:
                if not token in ['?','.','!',',']:
                    word_tokens.append(token)
        avgtoks = avgtoks + cntoks
        # write to proper row and column index
        df.loc[i,'Title'] = " ".join(word_tokens)
        vocab.extend(word_tokens)
        word_tokens.clear()  

        # also get each abstract as document in a list: towards TFIDF and tSNE
        abstract_as_documents.append(df.loc[i,'Title'])
        i = i + 1
    avgtoks = avgtoks/number_of_rows
    print("Average no. of tokens per row (Title): ", avgtoks)
    # also get each abstract as 'split' document in a list: towards Word2Vec and tSNE
    for doc in abstract_as_documents:
        abstract_as_split_documents.append(list(doc.split(" ")))

# ================================ TF-IDF =====================================

    # TFIDF tSNE Dimension Reduction & Visualization
    abstract_as_doc_tfidf_vector = TfidfVectorizer().fit_transform(abstract_as_documents)
    
    #Reduce the dimensions to 50 using SVD. The scientist in you can vary this!
    abstract_as_doc_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(abstract_as_doc_tfidf_vector)

    abs_normed_d2vec = np.zeros((number_of_rows,300))
    if not os.path.exists(cwd+'\\'+'dataset'+'\\'+'normed_doc_vector_{}'.format(number_of_rows) + '.npy'):

# ================================ Word2Vec; actually Doc2Vec =====================================
        #Note: - worker=1 and seeding to ensure reproducible results
        #      - tag vectors by patent 'Document_Number' or any unique identifier       
        tagged_abstract_as_split_documents = [TaggedDocument(doc, ['{}'.format(df.loc[i,'Document_Number'])]) 
                                            for i, doc in enumerate(abstract_as_split_documents)]
        # Intuition: docs are already treated for infrequent words, stop words etc; stave-off futher cuts by Doc2Vec with 
        #            min_count=1 default being 5             **truncation** avgtoks
        abstract_as_doc_d2vec = Doc2Vec(vector_size=300, window=20, seed=1234, workers=1, min_count=1, dm=0,
                                        alpha=0.25, min_alpha=0.0001, hs=0, negative=5)
        abstract_as_doc_d2vec.build_vocab(tagged_abstract_as_split_documents)
        #DBOW variant of Doc2Vec, so dm=1 ; learning rate decay with alpha & alpha min
        #hs=0 implies use neg.sampling; threfore negative samples to draw is defined by negative=?

        #Train the model
        abstract_as_doc_d2vec.train(tagged_abstract_as_split_documents, total_examples=abstract_as_doc_d2vec.corpus_count,
                                    epochs=500)#abstract_as_doc_d2vec.iter)
        
        #print("****DOC VECTORS***", abstract_as_doc_d2vec.docvecs.vectors_docs, '\n')
        #print('**** INDEXES ***', abstract_as_doc_d2vec.docvecs.index2entity)
        #print("****ONE DOC VEC: docvec[0]***", abstract_as_doc_d2vec.docvecs[ '{}'.format(df.loc[0,'Document_Number']) ] , '\n')

        # Normalization is necessary if we're using t-sne; does not do it internally but its QUITE efficient to.
        # Just as was done above for TF-IDF vector format.
        from sklearn.preprocessing import Normalizer
        nrm = Normalizer('l2') # L2 norm.
        normed = nrm.fit_transform(abstract_as_doc_d2vec.docvecs.vectors_docs)
        
        #Save the trained vectors! 
        np.save(cwd+'\\'+'dataset'+'\\'+'normed_doc_vector_{}'.format(number_of_rows) + '.npy', normed)
        ## Its dimension reduction, so the shape is changed. Get the new shape and re-asign new dimension-reduced features.
        abs_normed_d2vec.resize(normed.shape)
        abs_normed_d2vec = normed

        #Improve memory footprint; discard unwanted training metadata
        abstract_as_doc_d2vec.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=False)
    else:
        # Vector already exists from a previous run? Just load it then.
        abs_normed_d2vec = np.load(cwd+'\\'+'dataset'+'\\'+'normed_doc_vector_{}'.format(number_of_rows) + '.npy')
    
    # Run TSNE for either of TD-IDF or Word2Vec/Doc2Vec or both, if you wish
    """
    abstract_as_doc_embed = TSNE(n_components=3, perplexity=40, method='barnes_hut', verbose=2).fit_transform(abstract_as_doc_reduced) #init='pca'
    np.save(cwd+'\\'+'dataset'+'\\'+'abstract_as_doc_embed_tfidf_{}'.format(number_of_rows) + '.npy', abstract_as_doc_embed)
    display_word_cloud(abstract_as_doc_embed,df['Document_Number'])
    """
    # Important: Read-up on TSNE perplexity and the algorithms it deploys. The choice of components was arbitrary, to explain
    # the variance in the data. Let the scientist in you act out.
    abstract_as_doc_embed = TSNE(n_components=3, perplexity=50, method='barnes_hut', verbose=2).fit_transform(abs_normed_d2vec) #init='pca'
    np.save(cwd+'\\'+'dataset'+'\\'+'abstract_as_doc_embed_d2vec_{}'.format(number_of_rows) + '.npy', abstract_as_doc_embed)

    # Display TSNE for a visual of documents in latent-space, labelled by patent number i.e. 'Document_Number'.
    display_word_cloud(abstract_as_doc_embed, df['Document_Number'],2)
    
    # Write changes to file (and close)
    df.to_csv(cwd+'\\'+'dataset'+'\\'+'dataset.csv')

else:
    # A copy of the model should exist on drive from a previous run; try to restore proceedigs from there.
    pass

## Yes, I know, code should have been refactored. Well, get on with it yourself.


#      ======================    NOW TO K-CLUSTERING, but firstly: LOAD/BUILD VECTORS; READ UPDATED DATASET  ====================================

#open and read updated file contents
df = pd.read_csv(cwd+'\\'+'dataset'+'\\'+'dataset.csv', error_bad_lines=False).drop(['Unnamed: 0'], axis=1)
document_numbers = df['Document_Number'].tolist()
number_of_rows = len(df.index)

# The solling code section up until the 'else:' is not necessary. Just that a GLOVE mebdding can also be used.
# Show off! Bottomline, just  load the a vector i.e vec_lib = np.load(...)

GLOVE = False
if not os.path.exists(cwd+'\\'+'dataset'+'\\'+'abstract_as_doc_embed_d2vec_{}'.format(number_of_rows) + '.npy'):
    #Use spacy's GLOVE pretrained embeddings
    vec_lib = spacy.load('en_core_web_lg')   #the larger one is en_core_web_lg
    GLOVE = True
    #                               OR
    # the TFIDF file as the alternative... **Though both files could have been deleted?
else:
    vec_lib = np.load(cwd+'\\'+'dataset'+'\\'+'abstract_as_doc_embed_d2vec_{}'.format(number_of_rows) + '.npy')
    
# ==================================== K-MEANS CLUSTERING ==============================
## Randomly sample centroids
# set seed : 1234

#The experiments (i.e. finding optimal kK) we want conducted.
kays = [54,55,56,58,59,60,61,62,63]   # reads like '54 clusters, 60 iterations; 55 clusters, 60 iterations...'
iterant = [60,60,60,60,60,60,60,60,60]

distance_dict = {}
word_vector_dict = {}
centroid_dict = {}
old_centroids = {}
error_rate_over_iterations = {}
clusters = {}
clusters2 = {} # same but points are repped by their row number in the dataset
panel_error_rates = {} # A dashboard for ongoing errors
clusters_panel = {} # Holds final cluster allocations for each experiment
clusters_panel2 = {} #similarly but for format 2

for indx in range(len(kays)):
    k = kays[indx]
    iterations = iterant[indx]

    print('=== New Experiment ===', '\n')
    # Get k number of centroids randomly
    df_init_centroids = df.sample(n=k, random_state=1234, replace=False)
    #print('**init centroids**',df_init_centroids)
    init_centroids = df_init_centroids['Title'].tolist()#init_centroids = df_init_centroids['Abstract'].tolist()
    num_of_cents = len(init_centroids)

    # obtain the vector for each init centroid
    for i in range(num_of_cents):
        if GLOVE:
            centroid = vec_lib(init_centroids[i])
            c = np.reshape(np.array(centroid.vector), (1,300)) 
            # we can init old_centroids dict to zeros here
            old_centroids.update({i:np.zeros((1,300))})
        else:
            c = np.reshape(np.array(vec_lib[df_init_centroids.index[i]]), (1,3))
            old_centroids.update({i:np.zeros((1,3))})
        centroid_dict.update({i:c})

    temp1 = np.array(list(centroid_dict.values()))
    temp2 = np.array(list(old_centroids.values()))
    #Error (RSS)
    error = np.linalg.norm(temp2 - temp1)
    it = 1
    while it <= iterations:              #EARLY STOPPING 1 of entire process

        #panel_error_rates.update( { 'er_of_{}_{}'.format(k,iterations) : error_rate_over_iterations  } )
        # if we have convergence, then skip operations in remaining number of iterations
        if error!=0.0:
            print('Starting iteration...:', it, '  Error:', error )
            i = 0
            while i < num_of_cents:
                j = 0
                while j < number_of_rows: #number_of_rows being = to number of data points                    
                    if GLOVE:
                        point = vec_lib(df.loc[j,'Title']) #point = vec_lib(df.loc[j,'Abstract'])  <=== juts in case you like Abstracts rather than Title
                        p = np.reshape(np.array(point.vector), (1,300))
                    else:
                        p = np.reshape(vec_lib[j], (1,3))

                    #store the (doc_|word_)vectors indexed by row number, j
                    word_vector_dict.update({j:p})
                    # calculate the distance between centroid (in focus) and each point (in the decision region)
                    dm = cdist(centroid_dict[i], p, 'cosine')   #euclidean
                    if j not in distance_dict.keys():
                        distance_dict.update( { j:[ list(dm[0]) ] } )
                    else: 
                        distance_dict[j][0].append(dm[0][0])

                    if num_of_cents - i == 1 and number_of_rows - j == 1:
                        old_centroids = deepcopy(centroid_dict)

                        # add the index of argmin{distances} to dist dictionary
                        for m in range(number_of_rows):  
                            distance_dict[m].append(np.argmin(distance_dict[m][0]))
                        
                        # UPDATE THE CENTROIDS    
                        for n in range(num_of_cents):
                            count = 0
                            if GLOVE:
                                vector_sum = np.zeros((1,300),dtype=float)
                            else:
                                vector_sum = np.zeros((1,3),dtype=float)
                            for p in range(number_of_rows):
                                # check for allocations of points to each cluster
                                if distance_dict[p][1] == n:
                                    # calculate number of assignments to cluster n
                                    count = count + 1
                                    # We aim to take a mean of the points assigned to each cluster;
                                    # hence the count and summing (of vectors of each point in a cluster)
                                    vector_sum = np.array( np.add(word_vector_dict[p], vector_sum), dtype=float )
                            
                            # -- Check for div-by-zero; non-assigned clusters is a possibility! --
                            if count != 0:
                                updated_roid = np.true_divide(vector_sum,float(count))
                                if GLOVE:
                                    updated_roid = np.reshape(updated_roid, (1,300))
                                else:
                                    updated_roid = np.reshape(updated_roid, (1,3))
                                #distance_dict.update( {'updated_centroid_for_{}'.format(n):updated_roid[0] } )
                                centroid_dict[n] = updated_roid
                        
                        copy_distance_dict = deepcopy(distance_dict)
                        
                        # In-Experiment House Cleaning
                        for p in range(number_of_rows):
                            distance_dict[p][0].clear() 
                            distance_dict[p].pop() # --- next iteration then starts...
                            
                    j = j + 1
                i = i + 1
            # Compute residuals    
            temp1 = np.array(list(centroid_dict.values()))
            temp2 = np.array(list(old_centroids.values()))
            error = np.linalg.norm(temp2 - temp1)
            error_rate_over_iterations.update({it:error})
        else:
            print('Skipping ops in iteration...:', it, '  Error:', error )
            error_rate_over_iterations.update({it:error})
        
        if it == iterations:
            print("It\'s now equal to no. of iterations set")
            panel_error_rates.update( { 'er_of_{}_{}'.format(k,iterations) : error_rate_over_iterations  } )
            
        it = it + 1
    
    # THE FINAL CLUSTER ALLOCATIONS (per experiment)
    for t in copy_distance_dict.keys() :
        if copy_distance_dict[t][1] not in clusters.keys():
            clusters.update ( {copy_distance_dict[t][1] : [word_vector_dict[t]] } )
            clusters2.update( {copy_distance_dict[t][1]:[t] } )
        else: 
            clusters[copy_distance_dict[t][1]].append(word_vector_dict[t])
            clusters2[copy_distance_dict[t][1]].append(t)
    
    clusters_panel['er_of_{}_{}'.format(k,iterations)] = clusters
    clusters_panel2['er_of_{}_{}'.format(k,iterations)] = clusters2
    
    # write to file but sort by keys first
    od_clusters = collections.OrderedDict(sorted(clusters.items()))
    od_clusters2 = collections.OrderedDict(sorted(clusters2.items()))
    cdf = pd.DataFrame.from_dict(od_clusters, orient='index')
    cdf.to_csv(cwd +'\\'+'dataset'+'\\'+'Ordered_clusters_'+str(k)+'_'+str(iterations)+'.csv')
    cdf2 = pd.DataFrame.from_dict(od_clusters2, orient='index')
    cdf2.to_csv(cwd +'\\'+'dataset'+'\\'+'Ordered_clusters2_'+str(k)+'_'+str(iterations)+'.csv')

    distance_dict = {}
    word_vector_dict = {}
    centroid_dict = {}
    old_centroids = {}
    error_rate_over_iterations = {}
    clusters = {}
    clusters2 = {}

pdf = pd.DataFrame.from_dict(panel_error_rates, orient='index').fillna(value=0.0)
pdf.to_csv(cwd +'\\'+'dataset'+'\\'+'panel_error_rates'+'.csv')

clstrdf = pd.DataFrame.from_dict(clusters_panel, orient='index').fillna(value=0)
clstrdf.to_csv(cwd +'\\'+'dataset'+'\\'+'panel_clusters.csv')

clstrdf2 = pd.DataFrame.from_dict(clusters_panel2, orient='index').fillna(value=0)
clstrdf2.to_csv(cwd +'\\'+'dataset'+'\\'+'panel_clusters2.csv')

r = lambda: random.randint(0,255)

# Plot Error Panel ----------
fig = plt.figure()
x = np.array(pdf.columns.values.tolist())
experiments = np.array(pdf.index.values.tolist())
exp_server = lambda d, c: [ plt.plot(c,np.array(pdf.iloc[idx].values).flatten(),
                            label=exp, 
                            color='#{:02x}{:02x}{:02x}'.format(r(),r(),r())) 
                          for idx, exp in enumerate(d) ]
exp_server(experiments, x)
plt.title('Error Over Iterations - All Experiments')
plt.legend(loc=0)
plt.xlabel('Iterations')
plt.ylabel('Error - distance btw old and new \'roid')
#plt.show()
fig.savefig(cwd +'\\'+'dataset'+'\\'+'error_rates_composite.png')

# Plot Clusters Panel -----------
clusters3 = {}
x = clstrdf.columns.values.tolist()  # this is as long as the expr with the largest clusters
experiments = clstrdf.index.values.tolist()
# if it wont plot, write count of each allocation to file.....*SIGH*

counter = 0
Xs = []
Ys = []
Zs = []
list_of_trace = []

def plotter_writer(cursor, cee, dee):
    global counter
    global clusters3
    global Xs
    global Ys
    global Zs
    global list_of_trace
    e,c = cursor
    cXs = []
    cYs = []
    cZs = []
    counter +=1
    
    if not isinstance(clstrdf.iloc[e,c],(str,int,float)):
        #print("**** Cluster allocation exists****")
        length = len(clstrdf.iloc[e,c])
        #print("Number of points found:", length)
        temp = clstrdf.iloc[e,c]
        # go through the points in this cluster and access their x,y,z
        for j in range(length):
            x,y,z = np.array(np.reshape(temp.pop(), (3,))).tolist()
            Xs.append(x)
            Ys.append(y)
            Zs.append(z)
            cXs.append(x)
            cYs.append(y)
            cZs.append(z)
            # ....set label for plot to use here b4 exiting for...loop

        # ----Plot per cluster here-----
        # Add traces to list in prep for plotly plot
        list_of_trace.append( add_trace([cXs],[cYs],[cZs]) )
        cXs.clear
        cYs.clear
        cZs.clear
        clusters3.update({cee[c]:j+1})
    else:
        pass
    #File save per expr or plot Plotly for each experiment
    if counter == len(cee) and len(Xs)!= 0:
            file_path = cwd +'\\'+'dataset'+'\\'+'final_clusters_counts_{}'.format(dee[e])+'.csv'
            row_writer(file_path,clusters3) #pass in the header (cluster numbers) as well
            clusters3.clear()
            counter = 0
            # Plot on Plotly! Coool! # === alternatively plot 3D scatter  ======

            # 0 tells it to plot offline. WELL, DURRRH, YU NEED A PLOTLY A/C FOR THIS. Get one.
            # So, the code for the 'plotter' function has not been uploaded. It easy to use Plotly though. Sign-up
            plotter(list_of_trace,'{}'.format(dee[e]), 0)  
            list_of_trace.clear
            #for ita in range(len(Xs)):
            #    xyz(s[0][ita],t[0][ita],r[0][ita],'{}'.format(dee[e])) # Note: go.scatter3D expects x y z to be lists or other iterables hence the list wrapper 
         
    return

exp_server2 = lambda d, c: [ plotter_writer( (idx,idx2),c,d ) for idx, exp in enumerate(d) for idx2, col in enumerate(c) ]  
exp_server2(experiments,x)


del word_vector_dict
del centroid_dict
del old_centroids
del distance_dict
del copy_distance_dict
del error_rate_over_iterations
del clusters
del clusters2
del clusters3