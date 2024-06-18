import pandas as pd
from scipy.stats import  ttest_rel


def precision(type):
    map1 = 0  # Initialize map1 to 0, used for calculating Mean Average Precision (MAP)
    pre = {}  # Initialize an empty dictionary to store the precision for each query
    for query_id in range(101,151):  # For each query id from 101 to 150
        benFile = open(f'EvaluationBenchmark/Dataset{query_id}.txt')  # Open the corresponding benchmark file
        file_ = benFile.readlines()  # Read all lines from the file
        ben = {}  # Initialize an empty dictionary to store the data from the benchmark file
        for line in file_:  # For each line in the file
            line = line.strip()  # Remove trailing newline character
            lineList = line.split()  # Split the line into a list by spaces
            ben[lineList[1]] = float(lineList[2])  # Store the second element of the list as key and the third element as value (converted to float) in the dictionary
        benFile.close()  # Close the benchmark file
        rank1 = {}  # Initialize an empty dictionary to store the ranking
        i = 1  # Initialize a counter
        with open(f"RankingOutputs/{type}_R{query_id}Ranking.dat") as file:
            if type == "BM25":  # If type is "BM25"
                next(file)  # Skip the first line of the file
            for line in file:  # For each line in the file
                line = line.strip()  # Remove trailing newline character
                line1 = line.split()  # Split the line into a list by spaces
                rank1[str(i)] = line1[0]  # Store the counter (converted to string) as key and the first element of the list as value in the dictionary
                i = i + 1  # Increment the counter
        ri = 0  # Initialize ri to 0, used for counting the number of relevant documents
        avp = 0.0  # Initialize avp to 0.0, used for calculating Average Precision (AP)
        R = len([id for (id, v) in ben.items() if v > 0])  # Count the number of relevant documents in the benchmark file
        for (n, id) in sorted(rank1.items(), key=lambda x: int(x[0])):  # For each key-value pair in the dictionary sorted by key
            if (ben[id] > 0):  # If the document is relevant in the benchmark file
                ri = ri + 1  # Increment ri
                pi = float(ri) / float(int(n))  # Calculate precision
                avp = avp + pi  # Add precision to avp
        if ri == 0:  # If ri is 0, i.e., there are no relevant documents
            avp = 0  # Set avp to 0
        else:  # Otherwise
            avp = avp / float(ri)  # Calculate AP
        map1 = map1 + avp  # Add AP to MAP
        pre[query_id] = avp  # Store AP in the dictionary with query id as key
    map1 = map1/50  # Calculate MAP
    return map1, pre  # Return MAP and the precision for each query


if __name__ == "__main__":
    # Calculate the precision and MAP for each model
    map_JM, pre_JM = precision("JM_LM")
    map_bm25, pre_bm25 = precision("BM25")
    map_PRM, pre_PRM = precision("My_PRM")
    result_df = pd.DataFrame(columns=['Topic','BM25','JM_LM','MY_PRM'])
    # get the results from three models
    bm25 = list(pre_bm25.values())
    jm_lm = list(pre_JM.values())
    prm = list(pre_PRM.values())
    print(f' prm {map_PRM}')
    print(f' bm25 {map_bm25}')
    print(f' jm {map_JM}')
    # Do the t-test
    # test1: H0: jm_lm better than My_prm. H1: jm_lm worse than My_prm
    test1 = ttest_rel(prm, jm_lm, alternative='greater')
    # test2: H0: bm25 better than My_prm. H1: bm25 worse than My_prm
    test2 = ttest_rel(prm, bm25, alternative='greater')
    # test1: H0: My_prm better than jm_lm. H1: My_prm worse than jm_lm
    test3 = ttest_rel(jm_lm, prm, alternative='greater')
    # test1: H0: My_prm better than bm25. H1: My_prm worse than bm25
    test4 = ttest_rel(bm25, prm, alternative='greater')
    # test1: H0: jm_lm better than bm25. H1: jm_lm worse than bm25
    test5 = ttest_rel(bm25, jm_lm, alternative='greater')
    print(test1)
    print(test2)
    print(test3)
    print(test4)
    print(test5)



