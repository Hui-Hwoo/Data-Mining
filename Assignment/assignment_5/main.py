# Import everything, including homework code
import numpy as np  # numpy array
import pandas as pd  # data science
import matplotlib.pyplot as plt  # matplotlib
import seaborn as sns  # plot style
from sklearn.metrics import confusion_matrix  # confusion matrix
from sklearn.feature_extraction.text import CountVectorizer  # bag of words
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.preprocessing import MinMaxScaler  # scale data with min/max
from tqdm import tqdm  # download data utility
import json  # amazon data in json
from IPython.display import clear_output
from cs6220hw5 import clean_text
import time


# ========================================== #
#               Configurations               #
# ========================================== #
DATA_DIR = "data/"
OUTPUT_DIR = "output/"
Magazines_Path = f"{DATA_DIR}Magazine_Subscriptions.json"
Restaurant_Path = f"{DATA_DIR}Restaurant_Reviews.tsv"
TEST = False  # set to True to run the test
VOCAL_SIZE = 200
TEST_SIZE = 0.1
MINMAX_SCALE = True


# ========================================== #
#               Load Data                    #
# ========================================== #
import os

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load data in
reviews = []
count = 2
if TEST:
    data = pd.read_csv(Restaurant_Path, delimiter="\t", quoting=3)
else:
    with open(Magazines_Path, "r") as f:
        for l in tqdm(f):
            r = json.loads(l)
            reviews.append(r)
    # Format the data into Pandas DataFrame
    data = pd.DataFrame.from_records(reviews)[
        ["reviewText", "overall"]
    ]  # only keep reviewText and overall
    data = data[data["overall"] != 3.0]  # remove neutral reviews
    data = data.rename(columns={"reviewText": "Review"})
    data = data.dropna(subset=["Review"])  # remove data with NaN Review
    data["Liked"] = 0  # create a new column for liked and default to 0
    data.loc[data["overall"] > 3, "Liked"] = 1


# ========================================== #
#                 Histogram                  #
# ========================================== #

categories = ["Negative", "Positive"]
num_likes = np.sum(data["Liked"])
num_dislikes = data.shape[0] - num_likes

width = 0.3
positions = [0 + width / 2, 0.8 + width / 2]

# Plotting the bars
plt.clf()
plt.bar(positions, [0, num_likes], width=width, label="Negative")
plt.bar(positions, [num_dislikes, 0], width=width, label="Positive")

# Add labels and title
# add exact value for each bar
for i, v in enumerate([num_dislikes, num_likes]):
    plt.text(positions[i], v + 0.5, str(v), ha="center")
plt.xlabel("Reviews")
plt.ylabel("Frequency")
plt.title("Positive vs Negative Reviews")
plt.xticks(positions, categories)  # Shift x ticks to the right
plt.legend()

# Show plot
plt.tight_layout()
# plt.show()
plt.savefig(
    f"{OUTPUT_DIR}{'Restaurant' if TEST else 'Magazines'}_positive_vs_negative.png"
)


# ========================================== #
#               Preprocessing                #
# ========================================== #

# 0. Separate the data into training and testing
training_data, testing_data = train_test_split(
    data, test_size=TEST_SIZE, random_state=42
)

# 1. Rebalance
more_like = np.mean(training_data["Liked"]) > 0.5
minority_count = min(
    np.sum(training_data["Liked"]), len(training_data) - np.sum(training_data["Liked"])
)  # Determine the number of samples in the minority class

# Undersampling or Oversampling
minority = training_data[training_data["Liked"] == 1]
majority = training_data[training_data["Liked"] == 0]
if more_like:
    minority, majority = majority, minority

balanced_majority = majority.sample(n=minority_count, replace=False)  # Randomly select
balanced_data = pd.concat([balanced_majority, minority])

# 2. Clean up
print("Cleaning the data...")
start_time = time.time()
training_corpus, testing_corpus = clean_text(balanced_data, testing_data)
end_time = time.time()
print(f"Time to clean the data: {end_time - start_time} seconds")

# 3. Featurize
cv = CountVectorizer(max_features=VOCAL_SIZE)
x_train = cv.fit_transform(training_corpus).toarray()
y_train = np.array(balanced_data["Liked"])  # TODO: Change to balanced_dat

x_test = cv.fit_transform(testing_corpus).toarray()
y_test = np.array(testing_data["Liked"])

if MINMAX_SCALE:
    mm = MinMaxScaler()
    x_train = mm.fit_transform(x_train)
    x_test = mm.transform(x_test)

# save x_train, y_train, x_test, y_test
np.save(f"{OUTPUT_DIR}x_train.npy", x_train)
np.save(f"{OUTPUT_DIR}y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}x_test.npy", x_test)
np.save(f"{OUTPUT_DIR}y_test.npy", y_test)


# ========================================== #
#               Train Model                  #
# ========================================== #

# Naive Bayes, Random Forest, Decision Tree Classifier, Logistic Regression

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, auc

# Create a list of models
models = [
    ("GaussianNB", GaussianNB(var_smoothing=1e-9)),
    (
        "RandomForestClassifier",
        RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0),
    ),
    (
        "DecisionTreeClassifier",
        DecisionTreeClassifier(
            random_state=0,
            max_depth=2,
            min_samples_leaf=5,
            min_samples_split=5,
            max_features=2,
            criterion="gini",
        ),
    ),
    (
        "LogisticRegression",
        LogisticRegression(random_state=0, max_iter=1000, multi_class="ovr"),
    ),
    (
        "SVC",
        SVC(
            # C=1.0,
            # kernel='rbf',
            # degree=3,
            # gamma='scale',
            # coef0=0.0,
            # shrinking=True,
            # probability=False,
            # tol=0.001,
            # cache_size=200,
            # class_weight=None,
            # verbose=False,
            # max_iter=-1,
            # decision_function_shape='ovr',
            # break_ties=False,
            # random_state=None,
        ),
    ),
    ("MLPClassifier", MLPClassifier(max_iter=500, learning_rate_init=0.001)),
]

# read # save x_train, y_train, x_test, y_test from file
x_train = np.load(f"{OUTPUT_DIR}x_train.npy")
y_train = np.load(f"{OUTPUT_DIR}y_train.npy")
x_test = np.load(f"{OUTPUT_DIR}x_test.npy")
y_test = np.load(f"{OUTPUT_DIR}y_test.npy")

# Train each model
results = []
for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # Precision / Recall AUC
    precision_recall_curve(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    # Append the results
    results.append((name, accuracy, cm, pr_auc))


# ========================================== #
#               Results                      #
# ========================================== #
# Sort the results
results.sort(key=lambda x: x[1], reverse=True)

# Print out the results
for name, accuracy, cm, pr_auc in results:
    print(f"{name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix: {cm}")
    print(f"Precision-Recall AUC: {pr_auc:.2f}")
    print("\n")

    
    