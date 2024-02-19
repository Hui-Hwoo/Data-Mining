# =================================================== #
#                 CS 6200 Data Mining                 #
#                     Assignment 3                    #
#                Hui Hu (NUID: 002912425)             #
#                hu.hui1@northeastern.edu             #
# =================================================== #

import pyspark
from typing import Tuple, List
import os
import heapq

# =============================================== #
#                Pipeline Sketch                  #
# =============================================== #
"""
    Utilize spark to break dowm user and their friends into pairs
    and then count the number of mutual friends for each pair.
    Then, for each user, sort the pairs by the number of mutual friends, and output the top 10 pairs.
"""


# Set the Python hash seed
os.environ["PYTHONHASHSEED"] = "0"
sc = pyspark.SparkContext()

# start count time
import time
start_time = time.time()

# 1. Load the data
datapath = "data/soc-LiveJournal1Adj.txt"

lines = (
    sc.textFile(datapath, 1)
    .map(lambda line: line.split()).map(
        lambda x: (int(x[0]), [int(i) for i in x[1].split(",")] if len(x) > 1 else [])
    )
)

"""
key: user_id
value: friends list

[
    (0, [1, 2, 3, 4 ...]),
    ...
    (40000, [0, 2, 3, 4 ...]),
]
"""

# 2. Extract potential friends
# If A and B are recommended to be friends, they have a mutual friend C, then A and B will both in the list of C's friends. 
# Therefore, we can extract potential pairs of mutual friends from the list of friends of each user, and then count the frequency for each pair.

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
key: (user_id_1, user_id_2)
value: frequency 
(if they are already friends, then frequency is negative infinity)

((0, 1), -inf)                                                              
((0, 2), -inf)
((0, 4), -inf)
((1, 2), 1)
((1, 3), 1)
...
"""

# 3. Count the frequency for each pair
new_friend_pairs = (
    friend_pairs.reduceByKey(lambda x, y: x + y)
    .filter(lambda x: x[1] > 0)
    .flatMap(lambda x: [x, ((x[0][1], x[0][0]), x[1])])
)
"""
((1, 4), 2)                                                                 
((4, 1), 2)
((2, 3), 2)
((3, 2), 2)
((3, 4), 1)
((4, 3), 1)
"""

# 4. Extract potential friends with frequency for each user
potential_friends_with_frequency = new_friend_pairs.map(
    lambda x: (x[0][0], [(x[0][1], x[1])])
).reduceByKey(lambda x, y: x + y)

"""
key: user_id
value: list of (friend_id, frequency)

(1, [(4, 2)])                                                               
(4, [(1, 2), (3, 1)])
(2, [(3, 2)])
(3, [(2, 2), (4, 1)])
"""


# 5. Extract the top 10 potential friends for each user

def extract_n_friends_by_frequency(
    params: Tuple[int, List[Tuple[int, int]]], n: int = 10
):
    user_id, friends_list = params

    heap = []
    for index, value in enumerate(friends_list):
        if index < n:
            heapq.heappush(heap, (value[1], -value[0]))
        else:
            heapq.heappushpop(heap, (value[1], -value[0]))

    friends = []
    for _ in range(len(heap)):
        friends.insert(0, -heapq.heappop(heap)[1])

    return (user_id, list(friends))


potential_friends = potential_friends_with_frequency.map(extract_n_friends_by_frequency)
"""
key: user_id
value: top 10 friends with most mutual friends

(1, [4])                                                                    
(4, [1, 3])
(2, [3])
(3, [2, 4])
"""

# 6. format the output

output = potential_friends.map(lambda x: f"{x[0]}\t{','.join([str(i) for i in x[1]])}")
output.saveAsTextFile("output")

end_time = time.time()
print("Time elapsed: ", end_time - start_time)


# 7. Output the result
# user_list = [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]
# ans = potential_friends.filter(lambda x: x[0] in user_list).collect()
# for user in ans:
#     print(f"user {user[0]}", user[1])

"""
user 924 [439, 2409, 6995, 11860, 15416, 43748, 45881]
user 9020 [9021, 9016, 9017, 9022, 317, 9023]
user 9019 [9022, 317, 9023]
user 9993 [9991, 13134, 13478, 13877, 34299, 34485, 34642, 37941]
user 9022 [9019, 9020, 9021, 317, 9016, 9017, 9023]
user 8941 [8943, 8944, 8940]
user 9992 [9987, 9989, 35667, 9991]
user 9021 [9020, 9016, 9017, 9022, 317, 9023]
user 9990 [13134, 13478, 13877, 34299, 34485, 34642, 37941]
user 8942 [8939, 8940, 8943, 8944]
"""



