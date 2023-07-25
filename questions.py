import nltk
import sys
import os
import string
import math 
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)

    
def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_contents = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                contents = file.read()
                file_contents[filename] = contents
    return file_contents

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Convert document to lowercase
    document = document.lower()

    # Tokenize the document
    tokens = nltk.tokenize.word_tokenize(document)

    # Filter out punctuation and stopwords
    punctuation = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in punctuation and token not in stop_words]
    return filtered_tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    processed_words = []
    for file_key in documents:
        for word in set(documents[file_key]):
            if word not in processed_words:
                doc_count = -1 # Because we will loop over its current document as well
                for file_key_2 in documents:
                    if word in set(documents[file_key_2]):
                        doc_count += 1
                idfs[word] = math.log((len(documents) + 1)/(doc_count + 1)) # + 1 For smoothing
                processed_words.append(word)
    
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    scores = {}
    for file in files:
        score = 0
        for word in set(files[file]):
            if word in query:
                score += files[file].count(word) * idfs[word]
        scores[file] = score
    t_files = sorted(files, key = lambda k: scores[k], reverse=True)[:n]
    return t_files




def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = {}
    for sentence in sentences:
        matching_word_measure = 0
        query_term_density = 0

        for word in set(sentences[sentence]):
            if word in query:
                matching_word_measure += idfs[word]

        query_term_density = sum(1 for word in sentences[sentence] if word in query) / len(sentences[sentence])

        scores[sentence] = (matching_word_measure, query_term_density)

    top_n_sentences = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:n]
    return top_n_sentences

if __name__ == "__main__":
    main()
