OS: Windows 11
Python version: Python 3.10

Make sure you install the below packages:
os
string
math
stemming.porter2
re
pandas
ttest_rel from scipy.stats


Make sure the folder structure as below:
--Root Folder
---- Data_Collection
---- EvaluationBenchmark
---- RankingOutputs
-------- weights
---- common-english-words.txt
---- the50Queries.txt
---- DCG.py
---- MAP.py 
---- t_test.py
---- Precision_at10.py
---- Task1_to_Task3.py

The Data_Collection, EvaluationBenchmark folder and the50Queries.txt are the given data sets and Queries topic.
DCG.py,MAP.py ,t_test.py,Precision_at10.py and Task1_to_Task3.py. These 4 files are our model and evaluation function.   

While the "Task1_to_Task3.py" file is designed to create the "RankingOutputs" folder and the "weights" subfolder during execution, this might not always occur as expected. To ensure the script runs without any issues, you might need to manually create the "RankingOutputs" folder and the "weights" subfolder inside it. This extra step can help you avoid any potential errors and ensure the smooth execution of your project.

It is easy to run the project with PyCharm, simply create a project and then move all the above files and folders into it. Run the code in below order:
1, Task1_to_Task3.py
2, MAP.py 
3, Precision_at10.py
4, DCG.py
5, t_test.py

Using the command line in PowerShell:
You need to go to the Root Folder and open the folder with PowerShell. Then run the below command in order:
1, python Task1_to_Task3.py
2, python MAP.py
3, python DCG.py
4, python Precision_at10.py
5, python t_test.py

Please note that the Task1_to_Task3.py file takes some time to run due to the substantially large size of the document collection.
 
