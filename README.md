# test.ipynb — Experiment Index

---

## Experiment 1 — Text Preprocessing
**Cell:** 2

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
Core NLP preprocessing pipeline applied to 3 sample datasets. Covers sentence segmentation, word tokenization, stopword removal, stemming (Porter Stemmer), and lemmatization (WordNet). Includes a side-by-side comparison of original vs stemmed vs lemmatized tokens.

**TL;DR:** `sent_tokenize` · `word_tokenize` · `stopwords.words()` · `PorterStemmer().stem()` · `WordNetLemmatizer().lemmatize()`

---

## Experiment 2 — POS Tagging & Named Entity Recognition
**Cell:** 5

Part-of-speech tagging using NLTK's `pos_tag` and named entity recognition (NER) using spaCy's English model. Identifies entities such as persons, organizations, and locations from text.

**TL;DR:** `word_tokenize` · `pos_tag()` · `spacy.load("en_core_web_sm")` · `nlp(text).ents`

---

## Experiment 3 — Lexical Ambiguity Detection
**Cell:** 7

Detects lexically ambiguous words using WordNet. Tags words with POS, maps them to WordNet POS format, then retrieves and displays all possible synset definitions for ambiguous words in context.

**TL;DR:** `word_tokenize` · `pos_tag()` · `wordnet.synsets()` · `wordnet.lemmas()` · custom `get_wordnet_pos()`

---

## Experiment 3.2 — Lexical Ambiguity: Context Comparison
**Cell:** 8

Compares how an ambiguous word ("bank") carries different meanings across sentences using WordNet synsets. Illustrates the difference between financial and geographical senses of the same word.

**TL;DR:** `wordnet.synsets()` · `syn.definition()` · custom `compare_bank_meaning()`

---

## Experiment 3.3 — Syntactic Ambiguity Detection
**Cell:** 9

Detects syntactic ambiguity using a hand-written Context-Free Grammar (CFG) and NLTK's `ChartParser`. Generates multiple parse trees for ambiguous sentences (e.g., "I saw the man with the telescope") to show structural ambiguity.

**TL;DR:** `CFG.fromstring()` · `ChartParser()` · `parser.parse()` · `tree.pretty_print()`

---

## Experiment 4 – Part 1 — Term Frequency (TF)
**Cell:** 13

Calculates Term Frequency for words across multiple documents manually. Tokenizes text, counts word occurrences per document, and normalizes by document length to produce TF scores.

**TL;DR:** `str.lower().split()` · `Counter()` · manual TF formula: `count / total_words`

---

## Experiment 4 – Part 2 — Document Frequency (DF) & IDF
**Cell:** 14

Computes Document Frequency (how many documents contain each word) and Inverse Document Frequency (IDF = log(N/DF)). Identifies the highest and lowest IDF words (rare vs common terms).

**TL;DR:** `Counter()` · `set()` · `math.log()` · manual IDF formula: `log(N / df)`

---

## Experiment 4 – Part 3 — TF-IDF Matrix (scikit-learn)
**Cell:** 15

Builds a TF-IDF matrix using sklearn's `TfidfVectorizer`. Outputs the vocabulary and the full TF-IDF weight matrix across all documents.

**TL;DR:** `TfidfVectorizer()` · `fit_transform()` · `get_feature_names_out()` · `toarray()`

---

## Experiment 4 – Part 4 — Cosine Similarity
**Cells:** 16, 17

Computes pairwise cosine similarity between documents using the TF-IDF matrix. Displays the full similarity matrix, similarity scores relative to D1, and identifies the most similar document pair.

**TL;DR:** `cosine_similarity(tfidf_matrix)` · manual loop to find most similar pair

---

## Experiment 4 – Part 5 — Jaccard Similarity
**Cells:** 18, 19

Implements Jaccard Similarity from scratch using set intersection over union. Compares document pairs (D1 & D2, D1 & D3) based on shared unique tokens.

**TL;DR:** `set.intersection()` · `set.union()` · custom `jaccard_similarity()`: `|A∩B| / |A∪B|`

---

## Experiment 5 — Word2Vec Word Embeddings
**Cell:** 20

Loads a pre-trained Google News Word2Vec model (300-dimensional). Retrieves word vectors, finds the most similar words to a given word, and performs vector arithmetic (e.g., king − man + woman).

**TL;DR:** `KeyedVectors.load_word2vec_format()` · `model[word]` · `model.most_similar()` · vector arithmetic

---

## Experiment 6 — Text Classification
**Cell:** 28

Text classification on the BBC News dataset using TF-IDF features. Trains and evaluates two models — Multinomial Naïve Bayes and Logistic Regression — and compares their accuracy and classification reports.

**TL;DR:** `TfidfVectorizer()` · `train_test_split()` · `MultinomialNB()` · `LogisticRegression()` · `classification_report()` · `accuracy_score()`

---

## Experiment 9 — Sentiment Analysis
**Cell:** 37

Sentiment analysis on a review/tweet dataset. Uses VADER (NLTK) for rule-based sentiment scoring and Logistic Regression with TF-IDF for ML-based classification. Outputs accuracy, classification report, and a confusion matrix heatmap.

**TL;DR:** `SentimentIntensityAnalyzer()` · `sia.polarity_scores()` · `TfidfVectorizer()` · `LogisticRegression()` · `confusion_matrix()` · `sns.heatmap()`

---

> **Note:** Experiments 7 and 8 are not present in this notebook.
