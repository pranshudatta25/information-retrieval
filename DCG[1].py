import math
import os

import pandas as pd
from scipy.stats import ttest_ind

def read_relevance_judgments(folder_path):
    relevance_judgments = {} # Dictionary to store relevance judgments
    for file_name in os.listdir(folder_path): # Loop through each file in the folder
        if file_name.endswith('.txt'):  # Check if the file ends with '.txt'
            query_id = int(file_name.split('Dataset')[1].split('.')[0]) # Extract query ID from file name
            file_path = os.path.join(folder_path, file_name) # Construct file path
            with open(file_path, 'r') as file: # Open the file and read relevance judgments
                relevance_judgments[query_id] = {}  # Initialize sub-dictionary for each query ID
                for line in file: # Loop through each line in the file
                    query, doc_id, relevance = line.strip().split() # Split line into query ID, document ID, and relevance
                    relevance_judgments[query_id][doc_id] = int(relevance)  # Store relevance judgment in the dictionary
    return relevance_judgments

# Function to calculate DCG at k for a ranked list of documents
def calculate_dcg_at_k(ranked_list, relevance_judgments, k=10):
    dcg = 0 # Initialize DCG value
    for n, id in sorted(ranked_list.items(), key=lambda x: int(x[0])):  # Loop through each document in the ranked list
        if int(n)>k: # Check if position exceeds k
            break
        if int(n) == 1: # Calculate DCG value
            dcg = relevance_judgments[id]
        else:
            # print(n)
            dcg = dcg + (relevance_judgments[id]/math.log2(int(n)))
    # print(dcg)



    return dcg

# Function to evaluate models for queries
def evaluate_models_for_queries(model_names, relevance_judgments, ranking_folder):
    model_dcg10_scores = {model_name: {} for model_name in model_names} # Dictionary to store DCG@10 scores for each model

    for query_id in range(101, 151): # Loop through each query ID

        # print(rank1)
        for model_name in model_names: # Loop through each model
            rank1 = {} # Dictionary to store ranked list for each query
            i = 1 # Counter for rank
            ranking_file_path = os.path.join(ranking_folder, f"{model_name}_R{query_id}Ranking.dat")  # Construct path to ranking file
            with open(ranking_file_path, 'r') as file: # Open the ranking file
                if model_name == "BM25": # Skip header for BM25 model
                    next(file)
                for line in file:  # Loop through each line in the file
                    line = line.strip()
                    line1 = line.split()
                    rank1[str(i)] = line1[0]  # Store document ID in ranked list
                    i = i + 1
                dcg10 = calculate_dcg_at_k(rank1, relevance_judgments[query_id])  # Calculate DCG@10 for the ranked list
                model_dcg10_scores[model_name][query_id] = dcg10 # Store DCG@10 score for the model and query

    return model_dcg10_scores

# Function to print DCG@10 scores in a table format for different models
def print_dcg10_table_for_models(model_dcg10_scores, model_names):
    # print(model_dcg10_scores)
    print("Topic", end="\t")
    for model_name in model_names:
        print(model_name, end="\t")
    print()

    for query_id in range(101, 151): # Loop through each query ID
        print(f"R{query_id}", end="\t")
        for model_name in model_names: # Loop through each model
            score = float(model_dcg10_scores[model_name].get(query_id, '-'))
            print(f"{score:.3f}", end="\t")
        print()

    print("Average", end="\t")
    for model_name in model_names:
        avg_score = sum(model_dcg10_scores[model_name].values()) / len(model_dcg10_scores[model_name]) # Calculate average DCG@10 score for the model
        print(f"{avg_score:.3f}", end="\t")
    print()

# Function to calculate average DCG scores for each model
def average_dcg(model_dcg10_scores, model_names):
    avg = {} # Dictionary to store average DCG scores
    for model_name in model_names: # Loop through each model
        # Calculate average DCG score for the model
        avg_score = sum(model_dcg10_scores[model_name].values()) / len(model_dcg10_scores[model_name])
        avg[model_name] = avg_score # Store average DCG score in the dictionary
    return avg

if __name__ == "__main__":
    relevance_judgments = read_relevance_judgments("EvaluationBenchmark")  # Read relevance judgments
    model_names = ["BM25","JM_LM","My_PRM"] # Define model names
    model_dcg10_scores = evaluate_models_for_queries(model_names, relevance_judgments, "RankingOutputs")
    # print(model_dcg10_scores)
    bm25 = model_dcg10_scores['BM25']
    jm = model_dcg10_scores['JM_LM']
    prm = model_dcg10_scores['My_PRM']
    avg = average_dcg(model_dcg10_scores,model_names) # Calculate average DCG scores for each model
    avg_bm25 = avg['BM25']
    avg_jm = avg['JM_LM']
    avg_prm = avg['My_PRM']
    result_df = pd.DataFrame(columns=['Topic','BM25','JM_LM','MY_PRM'])


    # Function to insert DCG scores into the dataframe and calculate average
    def insert_pre(result_df, pre, map_value, column_name):
        for topic, value in pre.items(): # Loop through each topic and its DCG score
            if result_df[result_df['Topic'] == topic].empty:
                # Topic not in DataFrame, add new row
                temp_df = pd.DataFrame({'Topic': [topic], 'BM25': [None], 'JM_LM': [None], 'MY_PRM': [None]})
                temp_df[column_name] = value
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
            else:
                # Topic already in DataFrame, update the existing row
                result_df.loc[result_df['Topic'] == topic, column_name] = value

        # Add map value at the end of the column
        if result_df[result_df['Topic'] == 'Average'].empty:
            map_row = pd.DataFrame({'Topic': ['Average'], 'BM25': [None], 'JM_LM': [None], 'MY_PRM': [None]})
            map_row[column_name] = map_value
            result_df = pd.concat([result_df, map_row], ignore_index=True)
        else:
            result_df.loc[result_df['Topic'] == 'Average', column_name] = map_value
        return result_df
    result_df = insert_pre(result_df, bm25, avg_bm25, 'BM25')
    result_df = insert_pre(result_df, jm, avg_jm, 'JM_LM')
    result_df = insert_pre(result_df, prm, avg_prm, 'MY_PRM')
    print(result_df)
    result_df.to_csv("DCG.csv")