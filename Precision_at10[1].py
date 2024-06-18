## Evaluation

import os
import pandas as pd

#Read relevance judgments file provided to evaluate model effectiveness.

def read_relevance_judgments(folder_path):
    relevance_judgments = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            query_id = int(file_name.split('Dataset')[1].split('.')[0]) #Extract query id.
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                relevance_judgments[query_id] = {} #Store relevance flag as per document in topic.
                for line in file:
                    query, doc_id, relevance = line.strip().split()
                    relevance_judgments[query_id][doc_id] = int(relevance)
    return relevance_judgments


def calculate_precision_at_k(ranked_list, relevance_judgments, k=10):
    doc_relevant = 0
    precision = 0.0
    '''
    For top k (or, top 10) documents, if the relevance judgment benchmark file marks the document as relevant, add it to tally.
    '''
    for n, doc_id in sorted(ranked_list.items(), key=lambda x: int(x[0])):
        if int(n) > k:
            break
        if relevance_judgments.get(doc_id, 0) > 0:
            doc_relevant += 1
    #Tally of relevant documents from top k documents found in benchmark file divided by k.
    precision = doc_relevant / k
    return precision


def evaluate_models_for_queries(model_names, relevance_judgments, ranking_folder):
    model_precision10_scores = {model_name: {} for model_name in model_names}

    for query_id in range(101, 151):
        for model_name in model_names:
            rank1 = {}
            i = 1
            ranking_file_path = os.path.join(ranking_folder, f"{model_name}_R{query_id}Ranking.dat")
            with open(ranking_file_path, 'r') as file:
                if model_name == "BM25":
                    next(file)
                for line in file:
                    line = line.strip()
                    line1 = line.split()
                    rank1[str(i)] = line1[0]
                    i = i + 1
                precision10 = calculate_precision_at_k(rank1, relevance_judgments[query_id])
                model_precision10_scores[model_name][query_id] = precision10

    return model_precision10_scores


def print_precision10_table_for_models(model_precision10_scores, model_names):
    print("Topic", end="\t")
    for model_name in model_names:
        print(model_name, end="\t")
    print()

    for query_id in range(101, 151):
        print(f"R{query_id}", end="\t")
        for model_name in model_names:
            score = model_precision10_scores[model_name].get(query_id, '-')
            print(f"{score:.3f}", end="\t")
        print()

    print("Average", end="\t")
    for model_name in model_names:
        avg_score = sum(model_precision10_scores[model_name].values()) / len(model_precision10_scores[model_name])
        print(f"{avg_score:.3f}", end="\t")
    print()

def average_pre(model_dcg10_scores, model_names):
    avg = {}
    for model_name in model_names:
        avg_score = sum(model_dcg10_scores[model_name].values()) / len(model_dcg10_scores[model_name])
        avg[model_name] = avg_score
    return avg

if __name__ == "__main__":
    relevance_judgments = read_relevance_judgments("EvaluationBenchmark")
    model_names = ["BM25", "JM_LM", "My_PRM"]
    model_precision10_scores = evaluate_models_for_queries(model_names, relevance_judgments, "RankingOutputs")
    bm25 = model_precision10_scores['BM25']
    jm = model_precision10_scores['JM_LM']
    prm = model_precision10_scores['My_PRM']
    avg = average_pre(model_precision10_scores,model_names)
    avg_bm25 = avg['BM25']
    avg_jm = avg['JM_LM']
    avg_prm = avg['My_PRM']
    result_df = pd.DataFrame(columns=['Topic','BM25','JM_LM','MY_PRM'])
    def insert_pre(result_df, pre, map_value, column_name):
        for topic, value in pre.items():
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
    result_df.to_csv("Precision_at10.csv")
