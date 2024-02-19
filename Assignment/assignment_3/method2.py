import pyspark
from typing import Tuple, List
import os
import heapq

# Set the Python hash seed
os.environ["PYTHONHASHSEED"] = "0"
sc = pyspark.SparkContext()

datapath = "data/soc-LiveJournal1Adj-sample.txt"
datapath = "data/soc-LiveJournal1Adj.txt"
# datapath = "data/test.txt"

# start count time
import time

start_time = time.time()

# We pick the first 100 lines as a sample. Notice that IDs which are larger than 100 also show up in the friends list of the first 100 people.
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
        if user_id < friend:
            ans.append(((user_id, friend), float("-inf")))
        else:
            ans.append(((friend, user_id), float("-inf")))

    for i in range(n):
        for j in range(i + 1, n):
            if friends[i] < friends[j]:
                ans.append(((friends[i], friends[j]), 1))
            else:
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
new_friend_pairs = (
    friend_pairs.reduceByKey(lambda x, y: x + y)
    .filter(lambda x: x[1] > 0)
    .flatMap(lambda x: [x, ((x[0][1], x[0][0]), x[1])])
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

# check = potential_friends_with_frequency.map(lambda x: (x[0], len(x[1])))
# check.saveAsTextFile("check")


def extract_n_friends_by_frequency(
    params: Tuple[int, List[Tuple[int, int]]], n: int = 10
):
    user_id, friends_list = params

    # heap = []
    # for index, value in enumerate(friends_list):
    #     if index < n:
    #         heapq.heappush(heap, (value[1], value[0]))
    #     else:
    #         heapq.heappushpop(heap, (value[1], value[0]))

    # friends = [k[1] for k in heap]
    sorted_friends_list = sorted(
        friends_list, key=lambda x: (x[1], -int(x[0])), reverse=True
    )

    friends = [k[0] for i, k in enumerate(sorted_friends_list) if i < n]

    return (user_id, list(friends))


potential_friends = potential_friends_with_frequency.map(extract_n_friends_by_frequency)
"""
('1', ['4'])                                                                    
('4', ['1', '3'])
('2', ['3'])
('3', ['2', '4'])
"""


potential_friends.saveAsTextFile("method2")

end_time = time.time()
print("Time elapsed: ", end_time - start_time)

# res = potential_friends.take(10)
# for i in res:
#     print(i)

#  924,
# 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993.


# Time elapsed:  600.7353851795197 
