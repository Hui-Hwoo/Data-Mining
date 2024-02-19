import pyspark
from typing import Tuple, List
import os

# Set the Python hash seed
os.environ["PYTHONHASHSEED"] = "0"
sc = pyspark.SparkContext()

datapath = "data/soc-LiveJournal1Adj-sample.txt"
datapath = "data/soc-LiveJournal1Adj.txt"
# datapath = "data/test.txt"

# start count time
import time

start_time = time.time()

# @title We pick the first 100 lines as a sample. Notice that IDs which are larger than 100 also show up in the friends list of the first 100 people.
lines = (
    sc.textFile(datapath, 1)
    # .zipWithIndex()  # (line, index)
    # .filter(lambda x: x[1] < 10000)
    # .map(lambda x: x[0])
    .map(lambda line: line.split()).map(
        lambda x: (int(x[0]), [int(i) for i in x[1].split(",")] if len(x) > 1 else [])
    )
)


def extract_friend_pairs(params: Tuple[str, List[str]]):
    user_id, friends = params

    n = len(friends)
    ans = []

    for friend in friends:
        ans.append(((user_id, friend), float("-inf")))

    for i in range(n):
        for j in range(i + 1, n):
            ans.append(((friends[i], friends[j]), 1))
            ans.append(((friends[j], friends[i]), 1))

    return ans


friend_pairs = lines.flatMap(extract_friend_pairs)
"""
(('0', '1'), -inf)                                                              
(('0', '2'), -inf)
(('0', '3'), -inf)
(('0', '4'), -inf)
(('1', '2'), 1)
(('2', '1'), 1)
(('1', '3'), 1)
(('3', '1'), 1)
(('1', '4'), 1)
(('4', '1'), 1)
(('2', '3'), 1)
(('3', '2'), 1)
(('2', '4'), 1)
(('4', '2'), 1)
(('3', '4'), 1)
(('4', '3'), 1)
(('1', '0'), -inf)
(('1', '3'), -inf)
(('1', '2'), -inf)
(('0', '3'), 1)
(('3', '0'), 1)
(('0', '2'), 1)
(('2', '0'), 1)
(('3', '2'), 1)
(('2', '3'), 1)
(('2', '0'), -inf)
(('2', '1'), -inf)
(('2', '4'), -inf)
(('0', '1'), 1)
(('1', '0'), 1)
(('0', '4'), 1)
(('4', '0'), 1)
(('1', '4'), 1)
(('4', '1'), 1)
(('3', '0'), -inf)
(('3', '1'), -inf)
(('0', '1'), 1)
(('1', '0'), 1)
(('4', '0'), -inf)
(('4', '2'), -inf)
(('0', '2'), 1)
(('2', '0'), 1)
"""
new_friend_pairs = friend_pairs.reduceByKey(lambda x, y: x + y).filter(
    lambda x: x[1] > 0
)
"""
(('1', '4'), 2)                                                                 
(('4', '1'), 2)
(('2', '3'), 2)
(('3', '2'), 2)
(('3', '4'), 1)
(('4', '3'), 1)
"""

potential_friends_with_frequency = new_friend_pairs.map(
    lambda x: (x[0][0], [(x[0][1], x[1])])
).reduceByKey(lambda x, y: x + y)
"""
('1', [('4', 2)])                                                               
('4', [('1', 2), ('3', 1)])
('2', [('3', 2)])
('3', [('2', 2), ('4', 1)])
"""


def extract_n_friends_by_frequency(
    params: Tuple[str, List[Tuple[str, int]]], n: int = 10
):
    user_id, friends_list = params

    sorted_friends_list = sorted(
        friends_list, key=lambda x: (x[1], -int(x[0])), reverse=True
    )

    friends = [k[0] for i, k in enumerate(sorted_friends_list) if i < n]

    return (user_id, list(friends))


# res = potential_friends_with_frequency.map(lambda x: (x[0], sorted(x[1], key=lambda y: y[1], reverse=True)))

# for i in res.take(11):
#     print(i, end="\n\n")



potential_friends = potential_friends_with_frequency.map(extract_n_friends_by_frequency)
"""
('1', ['4'])                                                                    
('4', ['1', '3'])
('2', ['3'])
('3', ['2', '4'])
"""

potential_friends.saveAsTextFile("output")
# res = potential_friends.lookup("11")
# print(res)
# # for i in res:
# #     if i[0] == "11":
# #         print(i)

end_time = time.time()
print("Time elapsed: ", end_time - start_time)

"27552, 7785, 27573, 27574, 27589, 27590, 27600, 27617, 27620, 27667"
[
    [
        "27552",
        "7785",
        "27573",
        "27574",
        "27589",
        "27590",
        "27600",
        "27617",
        "27620",
        "27667",
    ]
]


#  924,
# 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993.
