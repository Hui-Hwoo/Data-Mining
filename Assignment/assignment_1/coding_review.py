def cardinality_items(filename: str = "data/basket_data.csv") -> int:
    """
    Takes a filename "*.csv" and returns an integer
    """
    # check file extension and existence
    if filename[-4:] != ".csv":
        raise ValueError("Invalid file extension: ", filename[-4:])
    try:
        hashset = set()
        with open(filename) as f:
            data = f.read()

        # read file line by line and split by comma
        for line in data.split("\n"):
            for item in line.split(","):
                # add item to hashset
                hashset.add(item.strip())
        # return the cardinality of the hashset
        print(filename, "Doneeeeee")
        return len(hashset)
    except FileNotFoundError:
        raise FileNotFoundError("File not found:", filename)

    return 0


def all_itemsets(items: list[str] = ["ham", "cheese", "bread"], k: int = 2) -> list[str]:
    """
    Takes a list of items and returns a list of all possible itemsets of size k
    """
    if k == 0:
        return [[]]
    if len(items) == 0:
        return []
    # get first item
    first = items[0]
    # get all itemsets of size k-1
    itemsets = all_itemsets(items[1:], k - 1)
    # add first to all itemsets
    for itemset in itemsets:
        itemset.insert(0, first)
    # get all itemsets of size k
    itemsets.extend(all_itemsets(items[1:], k))
    return itemsets


if __name__ == "__main__":
    print(cardinality_items())
    print(all_itemsets())
