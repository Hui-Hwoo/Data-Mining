# =================================================== #
#                 CS 6200 Data Mining                 #
#                     Assignment 1                    #
#                Hui Hu (NUID: 002912425)             #
#                hu.hui1@northeastern.edu             #
# =================================================== #

data_folder = "data/"
combined_data_list = [
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
]
movie_titles_file = "movie_titles.csv"


# ============================================== #
#          Review combined_data_*.txt            #
# ============================================== #


"""
1. How many total records of movie ratings are there in the entire dataset (over all of combined_data_*.txt)
2. How many total unique users are there in the entire dataset (over all of combined_data_*.txt)?
3. What is the range of years that this data is valid over?
"""


def get_num_records(data_folder=data_folder, file_list=combined_data_list):
    """
    :param file_list: list of file names
    :return: number of total records
    """
    num_records = 0
    for file in file_list:
        with open(f"{data_folder}/{file}", "r") as f:
            for line in f:
                if ":" not in line:
                    num_records += 1

    return num_records


def get_num_users(data_folder=data_folder, file_list=combined_data_list):
    """
    :param file_list: list of file names
    :return: number of total unique users
    """
    users = set()
    for file in file_list:
        with open(f"{data_folder}/{file}", "r") as f:
            for line in f:
                if ":" not in line:
                    users.add(line.split(",")[0].strip())

    return len(users)


def get_date_range(data_folder=data_folder, file_list=combined_data_list):
    """
    :param file_list: list of file names
    :return: range of years
    """
    start_date, end_date = "9999-99-99", "0000-00-00"
    for file in file_list:
        with open(f"{data_folder}/{file}", "r") as f:
            for line in f:
                if ":" not in line:
                    year = line.split(",")[2].strip()
                    if year < start_date:
                        start_date = year
                    elif year > end_date:
                        end_date = year

    return start_date, end_date


# ============================================== #
#           Review movie_titles.csv              #
# ============================================== #

"""
1. How many movies with unique names are there? That is to say, count the distinct names of the movies.
2. How many movie names refer to four different movies?
"""


def get_movie_pairs(data_folder=data_folder, movie_titles_file=movie_titles_file):
    """
    :param file: file name
    :return: number of movies
    """
    movie_pairs = {}
    with open(f"{data_folder}/{movie_titles_file}", "r", encoding="ISO-8859-1") as f:
        for line in f:
            movie_id, _, movie_name  = line.split(",", 2)
            movie_pairs[movie_id.strip()] = movie_name.strip()

    return movie_pairs


def get_num_movies(data_folder=data_folder, file=movie_titles_file):
    """
    :param file: file name
    :return: number of movies
    """
    movies = set()
    with open(f"{data_folder}/{file}", "r", encoding="ISO-8859-1") as f:
        for line in f:
            movies.add(line.split(",", 2)[2].strip())

    return len(movies)


def get_same_name_movies(data_folder=data_folder, file=movie_titles_file):
    """
    :param file: file name
    :return: number of movies with same name
    """
    movies = {}
    with open(f"{data_folder}/{file}", "r", encoding="ISO-8859-1") as f:
        for line in f:
            name = line.split(",", 2)[2].strip()
            if name in movies:
                movies[name] += 1
            else:
                movies[name] = 1

    count = 0
    for name in movies:
        if movies[name] == 4:
            count += 1

    return count


# ============================================== #
#                 Review Both                    #
# ============================================== #

"""
1. How many users rated exactly 200 movies?
2. Of these users, take the lowest user ID and print out the names of the movies that this person liked the most (all 5 star ratings).
"""


def get_users_200_movies(data_folder=data_folder, file_list=combined_data_list):
    """
    :param file_list: list of file names
    :return: list of users who rated exactly 200 movies
    """
    users = {}
    """
    users = {
        user_id1: {
            movies: {
                1: [movie1, movie2, ...], 
                2: [movie1, movie2, ...],
                 ...
                5: [movie1, movie2, ...]},
            }
            num_movies: 1
        },
        user_id2: {
            movies: { 
                1: [movie1, movie2, ...], 
                2: [movie1, movie2, ...],
                 ...
                5: [movie1, movie2, ...]},
            num_movies: 200
        },
        ...
    }

    """
    movie_id = None
    for file in file_list:
        with open(f"{data_folder}/{file}", "r") as f:
            for line in f:
                # get movie_id
                if ":" in line:
                    movie_id = line.split(":")[0].strip()
                else:
                    # get user_id rated movie_ID
                    user_id = line.split(",")[0].strip()
                    rate = int(line.split(",")[1].strip())

                    if user_id in users:
                        users[user_id]["num_movies"] += 1
                        users[user_id]["movies"][rate].append(movie_id)
                    else:
                        users[user_id] = {
                            "movies": {
                                1: [],
                                2: [],
                                3: [],
                                4: [],
                                5: [],
                            },
                            "num_movies": 1,
                        }
                        users[user_id]["movies"][rate].append(movie_id)

    users_with_200_rates = []
    for user_id in users.keys():
        if users[user_id]["num_movies"] == 200:
            users_with_200_rates.append((user_id, users[user_id]["movies"][5]))

    num_users = len(users_with_200_rates)
    lowest_user = min(users_with_200_rates, key=lambda x: int(x[0]))

    return num_users, lowest_user


def main():
    # ============================== #
    #   Review combined_data_*.txt   #
    # ============================== #
    print("\nStart to review combined_data_*.txt\n")

    # total records of movie ratings (100480507 records)
    num_records = get_num_records(data_folder=data_folder, file_list=combined_data_list)
    print("    Total records of movie ratings:", num_records)

    # total unique users (480189 users)
    num_users = get_num_users(data_folder=data_folder, file_list=combined_data_list)
    print("    Total unique users:", num_users)

    # range of years ('1999-11-11' to '2005-12-31')
    start_date, end_date = get_date_range(
        data_folder=data_folder, file_list=combined_data_list
    )
    print("    Range of years:", start_date, "to", end_date)

    # ============================== #
    #     Review movie_titles.csv    #
    # ============================== #

    print("\nStart to review movie_titles.csv\n")

    # number of movies (17297 movies)
    num_movies = get_num_movies(data_folder=data_folder, file=movie_titles_file)
    print("    Number of movies with unique names:", num_movies)

    # number of movies with same name (7 movies)
    num_same_name_movies = get_same_name_movies(
        data_folder=data_folder, file=movie_titles_file
    )
    print("    Number of movies with same name:", num_same_name_movies)

    # ============================== #
    #           Review Both          #
    # ============================== #

    print("\nStart to review both\n")

    # users who rated exactly 200 movies

    movie_pairs = get_movie_pairs(
        data_folder=data_folder, movie_titles_file=movie_titles_file
    )

    num_users, lowest_user = get_users_200_movies(
        data_folder=data_folder,
        file_list=combined_data_list
    )
    print("    Number of users who rated exactly 200 movies:", num_users)  # 605 users
    print("    Lowest user ID:", lowest_user[0])  # 1001192
    print("    Movies that this person liked the most:")
    for movie_id in lowest_user[1]:
        print(f"\t{movie_pairs[movie_id]}")


if __name__ == "__main__":
    main()
