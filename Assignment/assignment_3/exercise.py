import pyspark

sc = pyspark.SparkContext()

datapath = "data/soc-LiveJournal1Adj-sample.txt"


# @title We pick the first 100 lines as a sample. Notice that IDs which are larger than 100 also show up in the friends list of the first 100 people.
lines = (
    sc.textFile(datapath, 1)
    .zipWithIndex()
    .filter(lambda x: x[1] < 100)
    .map(lambda x: x[0])
    .map(lambda line: line.split())
)

# Display the number of friends for each person.
res1 = lines.map(lambda x: (x[0], len(x[1].split(",")))).collect()

# Show the IDs of those who have only four friends.
res2 = (
    lines.map(lambda x: (x[0], len(x[1].split(","))))
    .filter(lambda x: x[1] == 4)
    .collect()
)

# Who has the most friends?
res3 = (
    lines.map(lambda x: (x[0], len(x[1].split(","))))
    .sortBy(lambda x: x[1], ascending=False)
    .collect()[0]
)

# Display IDs categorized by the number of their friends.
res4 = (
    lines.map(lambda x: (len(x[1].split(",")), [x[0]]))
    .reduceByKey(lambda x, y: x + y)
    .collect()
)

# Find the average number of friends among all the people.
res5 = (
    lines.map(lambda x: (x[0], len(x[1].split(","))))
    .map(lambda x: ("-", (x[1], 1)))
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    .map(lambda x: x[1][0] / x[1][1])
    .collect()
)

# Show me the person that people are friends with the most.
"""
*Note, this isn't the person that has the most friends, but rather, the friend that has the most number of people having them as friends.
So, in the below data, it is not 1. It's 4, because 4 appears in everybody's friend list.
1 <tab> 2, 4, 5, 9, 12
2 <tab> 1, 4
3 <tab> 2, 3, 4, 6
4 <tab> 1, 2, 3, 4
"""
res6 = (
    lines.map(lambda x: x[1].split(","))
    .flatMap(lambda x: x)
    .map(lambda x: (x, 1))
    .reduceByKey(lambda x, y: x + y)
    .sortBy(lambda x: -x[1])
    .collect()[0][0]
)


# How many people are lone wolves that have no friends, but people are friends with them?
def fmt(x):
    y = []
    for i in x[1]:
        y.append((i, 1))
    return [(x[0], 0)] + y


res7 = (
    lines.map(lambda x: (x[0], x[1].split(",")))
    .map(lambda x: fmt(x))
    .flatMap(lambda x: x)
    .reduceByKey(lambda x, y: x * y)
    .filter(lambda x: x[1] == 1)
    .collect()
)

print(res7)