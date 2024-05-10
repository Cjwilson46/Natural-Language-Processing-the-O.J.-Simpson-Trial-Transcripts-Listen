import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist, pos_tag, word_tokenize

# Ensure the necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class SimpsonTrialAnalysis:
    def __init__(self, corpus_path):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"No such file or directory: {corpus_path}")
        
        self.corpus = PlaintextCorpusReader(corpus_path, '.*\.txt')
        self.tokens = nltk.word_tokenize(self.corpus.raw().lower())  # Convert to lowercase for consistent analysis
        self.text = nltk.Text(self.tokens)
        self.test_words = ['glove', 'gun', 'bronco', 'blood', 'guilty']

    def print_corpus_length(self):
        print(f"Corpus Length: {len(self.corpus.raw())}")

    def print_tokens_found(self):
        print(f"Number of Tokens Found: {len(self.tokens)}")

    def print_vocab_size(self):
        print(f"Vocabulary Size: {len(set(self.tokens))}")

    def print_word_occurrences(self):
        for word in self.test_words:
            print(f"Occurrences of {word}: {self.tokens.count(word)}")

    def print_concordance(self):
        for word in self.test_words:
            print(f"\nConcordance for {word}:")
            self.text.concordance(word, lines=5)

    def print_similarities(self):
        for word in self.test_words:
            print(f"\nSimilarities for {word}:")
            self.text.similar(word, num=20)

    def print_word_index(self):
        for word in self.test_words:
            try:
                index = self.tokens.index(word)
                print(f"First occurrence of {word} is at index: {index}")
            except ValueError:
                print(f"{word} not found in the corpus.")

if __name__ == "__main__":
    # This is the path to the corpus provided by you
    corpus_path = r'C:\Users\Administrator\Downloads\CORPUS'
    
    try:
        analyzer = SimpsonTrialAnalysis(corpus_path)
        # Running the analysis methods
        analyzer.print_corpus_length()
        analyzer.print_tokens_found()
        analyzer.print_vocab_size()
        analyzer.print_word_occurrences()
        analyzer.print_concordance()
        analyzer.print_similarities()
        analyzer.print_word_index()
    except FileNotFoundError as e:
        print(e)
