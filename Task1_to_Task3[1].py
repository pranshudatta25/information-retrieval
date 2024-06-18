import os
import string
import math
from stemming.porter2 import stem
import re


# Using this class to store the information of document, such as the length of the document
# The term frequency of the document
class DataCollection:
    def __init__(self, docID):
        self.docID = docID
        self.terms = {} # Dictionary to store terms and their frequencies
        self.doc_len = 0 # Variable to store the document length

    def add_doc_len(self):
        self.doc_len += 1 # Increment the document length by 1

    def add_term(self, term):
        try:
            self.terms[term] += 1 # Increment the term frequency by 1 if term exists
        except KeyError:
            self.terms[term] = 1  # Initialize the term frequency to 1 if term doesn't exist

    def get_term_list(self):
        return sorted(self.terms.items(), key=lambda x: x[1], reverse=True) # Return the terms sorted by frequency

    def get_doc_len(self):
        return self.doc_len # Return the document length

    def getDocID(self):
        return self.docID # Return the document ID


# Use this function to parse the document and store the document in the DataCollection class
def parse_collection(stop_words, inputpath):
    coll_docs = {}
    for files in os.listdir(inputpath): # Iterate over files in the input path
        if files.endswith(".xml"): # Process only XML files
            myfile = os.path.join(inputpath, files)
            file_ = open(myfile, 'r')
            docID = None
            start_end = False
            current_doc = None
            for line in file_:
                line = line.strip()
                if not start_end:
                    if line.startswith("<newsitem "):
                        for part in line.split():  # Get the item id of the document
                            if part.startswith("itemid="):
                                docID = part.split("=")[1].split("\"")[1]
                                break
                        current_doc = DataCollection(docID)
                    elif line.startswith("<text>"):
                        start_end = True
                elif line.startswith("</text>"):
                    break
                else:
                    # Process the main paragraphs of the document, get rid of the stopping words, punctuation and numbers
                    line = line.replace("<p>", "").replace("</p>", "")
                    line = line.translate(str.maketrans('', '', string.digits)).translate(
                        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    line = re.sub("\s+", " ", line)
                    for term in line.split():
                        term = stem(term.lower())
                        if len(term) > 2 and term not in stop_words:
                            current_doc.add_term(term)
                            current_doc.add_doc_len()
            if current_doc is not None:
                coll_docs[docID] = current_doc
    return coll_docs


# Calculate the average length
def avg_length(coll):
    totalDocLength = sum(doc.get_doc_len() for doc in coll.values()) # Calculate the total document length
    avgLength = totalDocLength / len(coll) # Calculate the average document length
    return avgLength


# In this project, the file common-english-words.txt is used as the collection of stopping words
stopwords_f = open('common-english-words.txt', 'r')
stopwordList = stopwords_f.read().split(',')


def my_bm25(coll, q, df):
    k1 = 1.2
    k2 = 500
    b = 0.75

    avgDocLength = avg_length(coll) # Calculate the average document length

    parsed_query = parse_query(q, stopwordList) # Parse the query

    bm25_score_dict = {} # Dictionary to store BM25 scores

    N = 2 * (len(coll))  # Adjusted for negative values

    for docID, doc in coll.items():
        bm25_score = 0 # Initialize BM25 score for the document

        for term, qf_w in parsed_query.items():
            if term in doc.terms:
                ni = df.get(term, 0) # Document frequency of the term
                fi = doc.terms[term] # Term frequency in the document
                x = math.log(((N - ni + 0.5) / (ni + 0.5)), 10) # IDF component
                K = (k1 * ((1 - b) + b * (doc.get_doc_len() / avgDocLength))) # Document length normalization
                y = ((fi * (k1 + 1)) / (K + fi)) # Term frequency normalization
                z = (((k2 + 1) * qf_w) / (k2 + qf_w)) # Query term frequency normalization

                bm25_term_score = x * (y * z) # Calculate BM25 score for the term

                bm25_score += bm25_term_score # Sum the BM25 score for the document

        bm25_score_dict[docID] = bm25_score # Store the BM25 score for the document

    return bm25_score_dict


def my_df(coll):
    df_ = {}
    for id, doc in coll.items():
        for term in doc.terms.keys():
            try:
                df_[term] += 1
            except KeyError:
                df_[term] = 1
    return df_


# Use this function to parse the query. In this project, the title of each query topic is used.
# Parse the query using the same method when parse the document.
# Get rid of the numbers, punctuations and stopping words.
def parse_query(query0, stop_words):
    terms = {}
    query0 = query0.replace("<p>", "").replace("</p>", "")
    query0 = query0.translate(str.maketrans('', '', string.digits)).translate(
        str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    query0 = re.sub("\s+", " ", query0)
    for term in query0.split():
        term = stem(term.lower())
        if len(term) > 2 and term not in stop_words:
            if term in terms:
                terms[term] += 1
            else:
                terms[term] = 1

    return terms


def w5(coll, ben, theta):
    T = {}
    # select T from relevant documents and r(tk)
    for id, doc in coll.items():
        if ben.get(id, 0) > 0:
            for term, freq in doc.terms.items():
                try:
                    T[term] += 1
                except KeyError:
                    T[term] = 1

    # calculate n(tk), number of documents the terms occur in the entire collection.
    ntk = {}
    for id, doc in coll.items():
        for term, freq in doc.terms.items():
            try:
                ntk[term] += 1
            except KeyError:
                ntk[term] = 1

    # calculate N and R
    No_docs = len(coll)
    R = 0
    for id, fre in ben.items():
        if ben[id] > 0:
            R += 1
     #r(tk) focuses on document term frequency for relevant documents.
     #n(tk) focuses on document term frequency for entire collection.
    for term, rtk in T.items():
        T[term] = ((rtk + 0.5) / (R - rtk + 0.5)) / (
                (ntk[term] - rtk + 0.5) / (No_docs - ntk[term] - R + rtk + 0.5))
    # calculate the mean of w5 weights.
    meanW5 = 0
    for id, rtk in T.items():
        meanW5 += rtk
    meanW5 = meanW5 / len(T)
    # Important features are identified and selected.
    Features = {t: r for t, r in T.items() if r > meanW5 + theta}
    return Features


def my_prm(coll, features):
    ranks = {}
    for id, doc in coll.items():
        Rank = 0
        #If a document contains the identified important features, the weight for that feature is added to the rank of that document.
        for term in features.keys():
            if term in doc.terms:
                try:
                    ranks[id] += features[term]
                except KeyError:
                    ranks[id] = features[term]
    return ranks


def jm_lm(coll, q, lambda_param):
    # Get the total length of all document
    N = sum(doc.get_doc_len() for doc in coll.values())
    # Parse the query string
    parsed_query = parse_query(q, stopwordList)
    jm_scores = {}
    for docID, doc in coll.items():
        # Initialize the score
        score = 1.0
        for term, qf in parsed_query.items():
            cqi = 0
            # Calculate the cqi, the appearance of the query terms in hole document set
            for d_id, d in coll.items():
                if term in d.terms:
                    cqi = cqi + d.terms[term]
            # If query terms not in the document, them set term frequency to 0
            if term not in doc.terms:
                term_freq = 0
            else:
                term_freq = doc.terms[term]
            # Calculate Jelinek-Mercer smoothed probability
            prob = (1 - lambda_param) * (term_freq / (doc.get_doc_len())) + lambda_param * (cqi / N)
            score = score * prob
        # Store the score in the dictionary
        jm_scores[docID] = score
    return jm_scores


if __name__ == "__main__":
    stopwords_f = open('common-english-words.txt', 'r')
    stopwordList = stopwords_f.read().split(',')
    inputpath = "Data_Collection"
    queries_file = "the50Queries.txt"
    output_folder = "RankingOutputs"
    weights_folder = os.path.join(output_folder, "weights")
    benchmark_file = 'PTraining_benchmark.txt'
    theta = 10.0  # Threshold for feature selection
    # Check if the folder exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
    # Read all the queries
    with open(queries_file, 'r') as file:
        queries = file.read().split('</Query>\n\n')

    with open(benchmark_file, 'w') as benchmark_write_file:
        benchmark_ = {}

        for query_block, query_id in zip(queries, range(101, 151)):
            if query_block.strip():
                # Parse query title
                query_title = ""
                query_lines = query_block.split('\n')
                for line in query_lines:
                    if line.startswith("<title>"):
                        query_title = line.replace("<title>", "").strip()
                        break

                data_folder = f"Data_C{query_id}"
                input_path = os.path.join(inputpath, data_folder)

                if not os.path.exists(input_path):
                    print(f"Data folder '{data_folder}' not found. Skipping query {query_id}...")
                    continue

                # Parse documents in data collection
                docs_collection = parse_collection(stopwordList, input_path)
                doc_freq_ = my_df(docs_collection)

                output_filename = f"BM25_R{query_id}Ranking.dat"
                output_path = os.path.join(output_folder, output_filename)
                output_filename_JM = f"JM_LM_R{query_id}Ranking.dat"
                output_path_JM = os.path.join(output_folder, output_filename_JM)
                with open(output_path_JM, "w") as f:
                    # Calculate JM_LM scores using query title
                    jm = jm_lm(docs_collection, query_title, lambda_param=0.4)
                    sorted_scores = sorted(jm.items(), key=lambda x: x[1], reverse=True)
                    for docID, score in sorted_scores:
                        doc = docs_collection[docID]
                        f.write(f"{docID} : {score}\n")
                with open(output_path, "w") as f:
                    f.write(f"Query {query_id} (DocID Weight):\n")
                    # Calculate BM25 scores using query title
                    bm25_scores = my_bm25(docs_collection, query_title, doc_freq_)
                    # Select the top 3 document as the relevant document and mark it as 1, others 0.
                    # This is done to create a benchmark file for My_PRM.
                    N = 3
                    assumed_relevant_docs = sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:N]
                    for doc_id, score in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{doc_id} {score}\n")
                        # If the document mark as relevant, set it to 1
                        if doc_id in assumed_relevant_docs:
                            relevance = 1
                        else:
                            relevance = 0
                        benchmark_write_file.write(f'R{query_id} {doc_id} {relevance}\n') #Store the relevance flag for corresponding topic

                        if f'R{query_id}' not in benchmark_:
                            benchmark_[f'R{query_id}'] = {}
                        benchmark_[f'R{query_id}'][doc_id] = relevance

        for query_block, query_id in zip(queries, range(101, 151)):
            if query_block.strip():
                data_folder = f"Data_C{query_id}"
                input_path = os.path.join(inputpath, data_folder)

                if not os.path.exists(input_path):
                    continue

                # Parse documents in data collection
                docs_collection = parse_collection(stopwordList, input_path)

                # Load the benchmarks calculated
                ben = benchmark_.get(f"R{query_id}", {})

                # Feature weights using w5 function
                feature_weights = w5(docs_collection, ben, theta)

                #Store weights in relevant topics' file.
                weights_file_path = os.path.join(weights_folder, f'PModel_w5_R{query_id}.dat')
                with open(weights_file_path, 'w') as wFile:
                    for (k, v) in sorted(feature_weights.items(), key=lambda x: x[1], reverse=True):
                        wFile.write(k + ' ' + str(v) + '\n')

                # Re-rank the documents using the calculated feature weights
                PRM_scores = my_prm(docs_collection, feature_weights)

                #Store results of pseudo-relevance model.
                PRM_output_path = os.path.join(output_folder, f"My_PRM_R{query_id}Ranking.dat")
                with open(PRM_output_path, "w") as f:
                    for doc_id, score in sorted(PRM_scores.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{doc_id} {score}\n")
