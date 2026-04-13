# Text preprocessing and normalization utilities

import re
from typing import List
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# Define a class for text preprocessing
class PreprocessText:
    """Text preprocessing pipeline."""
    
    # Initialize with option to remove stopwords
    # Default is to NOT remove stopwords bc skills with short tokens like "AI" or "R" may be removed 
    # Add remove_stopwords=True to remove stopwords during tokenization and lemmatization
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.nlp = nlp
    
    # Clean text by removing URLs, emails, special characters, and normalizing whitespace
    def clean_text(self, text: str) -> str:
        """
        Clean text: remove URLs, emails, special characters.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # remove special characters, keep alphanumeric and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\+\-\#]', ' ', text)
        # normalize whitespace
        text = ' '.join(text.split())
        
        return text.lower().strip()
    
    # tokenize text into words, optionally removing stopwords
    def tokenize(self, text: str) -> List[str]:
        """ 
        Tokenize text into words. 
        
        Args: 
            text: Input text 

        Returns: 
            List of tokens 
        """
        # clean the text before tokenization
        text = self.clean_text(text)
        tokens = word_tokenize(text)

        # don't remove these important short tokens
        IMPORTANT_SHORT_TOKENS = {"ai", "ml", "r", "c", "c++"}

        if self.remove_stopwords:
            cleaned_tokens = []
            for token in tokens:
                token = token.lower().strip()
                # skip empty tokens
                if not token:
                    continue
                # skip stopwords
                if token in self.stop_words:
                    continue
                # skip very short tokens unless important
                if len(token) <= 2 and token not in IMPORTANT_SHORT_TOKENS:
                    continue
                cleaned_tokens.append(token)

            tokens = cleaned_tokens

        return tokens
    
    # Lemmatize text using spaCy, optionally removing stopwords
    def lemmatize(self, text: str) -> List[str]:
        """
        Lemmatize text using spaCy.
        
        Args:
            text: Input text
        
        Returns:
            List of lemmatized tokens
        """
        # clean the text before lemmatization
        text = self.clean_text(text)

        # if spaCy model is not available, use tokenization without lemmatization
        if self.nlp is None:
            return self.tokenize(text)

        IMPORTANT_SHORT_TOKENS = {"ai", "ml", "r", "c", "c++"}

        doc = self.nlp(text)
        tokens = []

        for token in doc:
            lemma = token.lemma_.lower().strip()
            # skip punctuation and spaces
            if token.is_punct or not lemma:
                continue
            # skip stopwords
            if self.remove_stopwords and lemma in self.stop_words:
                continue
            # skip short tokens unless important
            if len(lemma) <= 2 and lemma not in IMPORTANT_SHORT_TOKENS:
                continue

            tokens.append(lemma)

        return tokens
    
    def process(self, text: str, method: str = 'lemmatize') -> List[str]:
        """
        Process text with specified method.
        
        Args:
            text: Input text
            method: 'clean', 'tokenize', or 'lemmatize'
        
        Returns:
            Processed tokens or cleaned text
        """
        if method == 'clean':
            return self.clean_text(text)
        elif method == 'tokenize':
            return self.tokenize(text)
        elif method == 'lemmatize':
            return self.lemmatize(text)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def process_batch(self, texts: List[str], method: str = 'lemmatize') -> List:
        """
        Process multiple texts.
        
        Args:
            texts: List of text strings
            method: Processing method
        
        Returns:
            List of processed texts
        """
        return [self.process(text, method) for text in texts]


# Quick function calls
def clean_text(text: str) -> str:
    """Quick text cleaning."""
    preprocessor = PreprocessText()
    return preprocessor.clean_text(text)

def tokenize(text: str) -> List[str]:
    """Quick tokenization."""
    preprocessor = PreprocessText()
    return preprocessor.tokenize(text)

def lemmatize(text: str) -> List[str]:
    """Quick lemmatization."""
    preprocessor = PreprocessText()
    return preprocessor.lemmatize(text)