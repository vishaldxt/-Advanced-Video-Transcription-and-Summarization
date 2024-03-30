from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Function to read files and compute similarity
def read_and_compare(file1, file2):
    with open(file1, 'r', encoding='utf-8') as file:
        text1 = file.read()
    with open(file2, 'r', encoding='utf-8') as file:
        text2 = file.read()
    return text1, text2

# Vectorize the text and compute the cosine similarity
def compute_cosine_similarity(text1, text2):
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# Function to calculate Jaccard similarity for entire documents
def calculate_jaccard_similarity(doc1, doc2):
    # Split the documents into sets of words
    words_doc1 = set(doc1.split())
    words_doc2 = set(doc2.split())
    # Calculate Jaccard similarity score
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)
    jaccard_score = len(intersection) / len(union)
    return jaccard_score

# Directory paths
test_dir_path = 'C:\\Users\\SachinSlade\\Desktop\\Test\\Content\\Test'
result_dir_path = 'C:\\Users\\SachinSlade\\Desktop\\Test\\Content\\Result'

# Assuming that the file names are in the format 'test(1).txt' to 'test(7).txt' and 'result(1).txt' to 'result(7).txt'
file_numbers = range(1, 8)

# Dictionary to hold both cosine and Jaccard similarity scores
combined_similarity_scores = {}

# Compute both similarities for each pair of test and result files
for i in file_numbers:
    test_file = os.path.join(test_dir_path, f"test({i}).txt")
    result_file = os.path.join(result_dir_path, f"result({i}).txt")
    text1, text2 = read_and_compare(test_file, result_file)
    cosine_score = compute_cosine_similarity(text1, text2)
    jaccard_score = calculate_jaccard_similarity(text1, text2)
    combined_similarity_scores[i] = {
        'Cosine Similarity': cosine_score,
        'Jaccard Similarity': jaccard_score
    }

# Output the combined similarity scores
for pair, scores in combined_similarity_scores.items():
    print(f"Pair {pair}: Cosine Similarity - {scores['Cosine Similarity']:.2%}, "
          f"Jaccard Similarity - {scores['Jaccard Similarity']:.2%}")
