import pandas as pd


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
    # I use this function to store the result in a data frame, so that I can save the result in a csv files
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
        if result_df[result_df['Topic'] == 'MAP'].empty:
            map_row = pd.DataFrame({'Topic': ['MAP'], 'BM25': [None], 'JM_LM': [None], 'MY_PRM': [None]})
            map_row[column_name] = map_value
            result_df = pd.concat([result_df, map_row], ignore_index=True)
        else:
            result_df.loc[result_df['Topic'] == 'MAP', column_name] = map_value
        return result_df
    result_df = insert_pre(result_df, pre_bm25, map_bm25, 'BM25')
    result_df = insert_pre(result_df, pre_JM, map_JM, 'JM_LM')
    result_df = insert_pre(result_df, pre_PRM, map_PRM, 'MY_PRM')
    print(result_df)
    result_df.to_csv("MAP.csv")


