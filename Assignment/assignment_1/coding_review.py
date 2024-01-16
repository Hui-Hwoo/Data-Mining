# =================================================== #
#                 CS 6200 Data Mining                 #
#                     Assignment 1                    #
#                Hui Hu (NUID: 002912425)             #
#                hu.hui1@northeastern.edu             #
# =================================================== #

data_folder = "data/"
filename = "basket_data.csv"


def cardinality_items(data_folder=data_folder, filename=filename) -> int:
    """
    Takes a filename "*.csv" and returns an integer
    """
    # check file extension and existence
    if filename[-4:] != ".csv":
        print("Invalid file extension: ", filename[-4:])
        return 0
    try:
        hashset = set()
        with open(f"{data_folder}/{filename}") as f:
            data = f.read()

        # read file line by line and split by comma
        for line in data.split("\n"):
            for item in line.split(","):
                # add item to hashset
                hashset.add(item.strip())

        # return the cardinality of the hashset
        return len(hashset)
    except FileNotFoundError:
        print("File not found:", filename)
        return 0


def all_itemsets(items: ["ham", "cheese", "bread"], k: int = 2) -> list[str]:
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


def main():
    cardinality = cardinality_items(data_folder=data_folder, filename=filename)
    print(f"Cardinality of items in {filename}: {cardinality}")

    item = ["ham", "cheese", "bread"]
    k = 2
    print(f"All itemsets of size {k} in {item}:", all_itemsets(item, k))


if __name__ == "__main__":
    main()
