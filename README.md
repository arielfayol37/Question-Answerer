# Question Answering System with Inverse Document Frequency

## Background

Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. In this project, we've designed a simple question answering system based on inverse document frequency.

Our question answering system performs two tasks: document retrieval and passage retrieval. We have access to a corpus of text documents, and when presented with a query (a question in English asked by the user), document retrieval identifies which document(s) are most relevant to the query. Once the top documents are found, they are subdivided into passages (in this case, sentences) to determine the most relevant passage to the question.

To find the most relevant documents, we use tf-idf to rank documents based on term frequency for words in the query as well as inverse document frequency for words in the query. For passage retrieval, we use a combination of inverse document frequency and a query term density measure.

## Understanding

First, take a look at the documents in the `corpus` directory. Each text file contains the contents of a Wikipedia page. The goal of our AI is to find sentences from these files that are relevant to a user’s query. Feel free to experiment with adding, removing, or modifying files in the corpus to work with a different set of documents.

Next, let's examine the `questions.py` script. The global variables `FILE_MATCHES` and `SENTENCES_MATCHES` specify how many files and sentences, respectively, should be matched for any given query. By default, each of these values is 1, meaning our AI will find the top sentence from the top matching document as the answer to the question. You can experiment with changing these values.

In the `main` function, we first load the files from the `corpus` directory into memory using the `load_files` function. Each file is then tokenized into a list of words, allowing us to compute inverse document frequency (IDF) values for each word using the `compute_idfs` function. The user is then prompted to enter a query.

The `top_files` function identifies the files that are the best match for the query. From those files, sentences are extracted, and the `top_sentences` function identifies the sentences that are the best match for the query.

Let's take a closer look at the functions that need implementation: `load_files`, `tokenize`, `compute_idfs`, `top_files`, and `top_sentences`.

### Specification

1. The `load_files` function accepts the name of a directory and returns a dictionary mapping the filename of each `.txt` file inside that directory to the file’s contents as a string.

2. The `tokenize` function accepts a document (a string) as input and returns a list of all the words in that document, in order and lowercased.

3. The `compute_idfs` function accepts a dictionary of documents and retursn a new dictionary mapping words to their IDF (inverse document frequency) values.

4. The `top_files` function, given a query (a set of words), files (a dictionary mapping filenames to lists of words), and IDFs (a dictionary mapping words to their IDF values), returns a list of the filenames of the `n` top files that match the query, ranked according to tf-idf.

5. The `top_sentences` function, given a query (a set of words), sentences (a dictionary mapping sentences to lists of words), and IDFs (a dictionary mapping words to their IDF values), returns a list of the `n` top sentences that match the query, ranked according to IDF.

With these functions implemented, our question answering system is complete! It can perform document retrieval and passage retrieval using inverse document frequency, allowing it to find the most relevant sentences to a user’s query from a corpus of text documents.
