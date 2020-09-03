#TDDE16_Project_Clustering_of_Patent_Documents

This project showcases exploratory data analysis. In particular, the aim was to cluster documents in a patents database represented as document 
embedding vectors, by way of K-Means clustering. For that purpose, a publicly available sample of the European Patents Office (EPO) DOCDB format
served as the source of data. Data pre-processing efforts included dropping unwanted XML tags, extracting the text of claims and further processing
to filter stop-words and punctuation. Document embedding vectors were created from the resulting corpus with Google’s Doc2Vec. These were then 
mapped to 3D space using T-distributed Stochastic Neighbor Embedding (tSNE), normalized (L2) and fed to K-Means algorithm. Visualization was by
Plotly.