# Evaluation

import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
import pyphen
import torch
import re
from transformers import pipeline

nltk.download('brown', quiet=True)
nltk.download('stopwords', quiet=True)

model_name = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
keybert_model = KeyBERT('distilbert-base-nli-mean-tokens')
dic = pyphen.Pyphen(lang='en')
zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def calculate_flesch_reading_ease(text):
    # Remove punctuation from the text
    text = re.sub(r'[^\w\s]', '', text)

    # Split the text into sentences and words
    sentences = re.split(r'[.!?]', text)
    words = re.findall(r'\w+', text)

    # Calculate the average number of words per sentence
    average_words_per_sentence = len(words) / len(sentences)

    # Calculate the average number of syllables per word
    total_syllables = 0
    for word in words:
        total_syllables += count_syllables(word)
    average_syllables_per_word = total_syllables / len(words)

    # Calculate the Flesch Reading Ease score
    fre_score = 206.835 - (1.015 * average_words_per_sentence) - (84.6 * average_syllables_per_word)

    return fre_score


def count_syllables(word):
    hyphenated_word = dic.inserted(word)
    syllables = hyphenated_word.split('-')
    return len(syllables)

import re
from collections import Counter
from nltk.corpus import stopwords

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove newline characters
    text = re.sub(r'\n', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def sentiment_alignment(text, summary):
    text = preprocess_text(text)
    summary = preprocess_text(summary)
        
    # Define the class names
    class_names = ["positive", "negative", "mixed"]
    
    # Get sentiment scores using the BART-large-mnli model
    text_sentiment = zero_shot_pipeline(text, class_names)
    text_output = text_sentiment['labels'][text_sentiment['scores'].index(max(text_sentiment['scores']))]
    
    summary_sentiment = zero_shot_pipeline(summary, class_names)
    summary_output = summary_sentiment['labels'][summary_sentiment['scores'].index(max(summary_sentiment['scores']))]
    
    if text_output == summary_output:
        return 1
    else:
        return 0


def detect_repetitive_phrases(input_string):
    stopwords_set = set(stopwords.words('english'))
    input_string = input_string.lower()
    words = re.findall(r'\b\w+\b', input_string)
    words = [word for word in words if word not in stopwords_set]
    phrases = [' '.join(t) for t in zip(words[:-2], words[1:-1], words[2:])]
    phrase_counts = Counter(phrases)
    repetitive_phrases = {phrase: count for phrase, count in phrase_counts.items() if count > 1}

    if repetitive_phrases:    # if dictionary is not empty
        return 0    # text contains repetitive phrases
    else:
        return 1    # no repetitive phrases found


def keyword_overlap(text_a, text_b):
    # Preprocess the text
    text_a = preprocess_text(text_a)
    text_b = preprocess_text(text_b)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    overlap_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return overlap_score
    

def sentence_similarity(sent_a, sent_b):
    # Tokenize the input sentences
    inputs = tokenizer([sent_a, sent_b], return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Get the embeddings from the MiniLM model
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # Calculate the similarity score
    similarity_score = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
    return similarity_score


def keyword_overlap(text_a, text_b):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    overlap_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return overlap_score


def readability(text):
    preproc_text = preprocess_text(text)
    return calculate_flesch_reading_ease(preproc_text)


def count_noun_phrases(text):
    text = preprocess_text(text)
    blob = TextBlob(text)
    return blob.noun_phrases


def feature_highlighting(text, summary):
    text = preprocess_text(text)
    summary = preprocess_text(summary)
    
    # Extract keywords from the text and summary using KeyBERT
    text_keywords = set([kw[0] for kw in keybert_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english')])
    summary_keywords = set([kw[0] for kw in keybert_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words='english')])
    
    # If both the text and summary have no keywords, return a score of 0
    if not text_keywords and not summary_keywords:
        return 0
    
    # Calculate the feature coverage score
    feature_coverage_score = len(text_keywords.intersection(summary_keywords)) / len(text_keywords)
    
    return feature_coverage_score


def extract_last_message(conversation):
    # Split the conversation string into individual messages
    messages = re.split(r'\nUser:', conversation)
    
    # Remove any leading or trailing whitespace from each message
    messages = [msg.strip() for msg in messages]
    
    # Identify the last message in the list of messages
    last_message = messages[-1]
    
    return last_message


def last_message_coverage(conversation, last_message, summary):
    return sentence_similarity(last_message, summary)


def context_inclusion(conversation, last_message, summary):
    conv_similarity = sentence_similarity(conversation, summary)
    last_msg_similarity = sentence_similarity(last_message, summary)
    return conv_similarity / last_msg_similarity


def pii_detection(summary, pii_list=[]):
    # Check for Twitter handles
    twitter_handle_pattern = r'@\w+'
    twitter_handles = re.findall(twitter_handle_pattern, summary)
    
    # Combine the PII list with the detected Twitter handles
    combined_pii_list = pii_list + twitter_handles
    
    # Check if any PII elements are present in the summary
    pii_present = any(pii in summary for pii in combined_pii_list)
    
    # Return 1 if no PII elements are present, 0 otherwise
    return 1 if not pii_present else 0


def product_name_presence(product_name, text, summary):
    if summary.lower() == "none":
        return 1
    in_text = product_name.lower() in text.lower()
    in_summary = product_name.lower() in summary.lower()
    return int(in_text == in_summary)

def evaluate_review_summary(product_name, text, summary):
    # Metrics for RecordTypeReview
    summary = preprocess_text(summary)
    
    coverage_score = keyword_overlap(text, summary)
    repitition_score = detect_repetitive_phrases(summary)
    text_sentences = len(preprocess_text(text).split('.'))
    summary_sentences = len(summary.split('.'))
    summary_to_text_ratio = summary_sentences / text_sentences
    conciseness_threshold = 0.7
    conciseness_score = 1 if summary_to_text_ratio <= conciseness_threshold else 0
    
    raw_clarity_score = max(0, calculate_flesch_reading_ease(summary)) if summary.lower() != "none" else 0  # Set a lower limit of 0 for clarity_score
    clarity_score = raw_clarity_score / 206.835
    sentiment_align_score = sentiment_alignment(text, summary)
    feature_highlight_score = feature_highlighting(text, summary)
    
    # Weights for RecordTypeReview
    weights = {
        "coverage": 0.1,
        "repitition": 0.1,
        "conciseness": 0.1,
        "clarity": 0.05,
        "sentiment_alignment": 0.2,
        "feature_highlighting": 0.2,
    }
    
    # Calculate the final evaluation score
    overall_score = (
        coverage_score * weights["coverage"]
        + repitition_score * weights["repitition"]
        + conciseness_score * weights["conciseness"]
        #+ clarity_score * weights["clarity"]
        + sentiment_align_score * weights["sentiment_alignment"]
        + feature_highlight_score * weights["feature_highlighting"]
    )
    return {
        "coverage_score": coverage_score,
        "repitition_score": repitition_score,
        #"clarity_score": clarity_score,
        "conciseness_score": conciseness_score,
        "sentiment_align_score": sentiment_align_score,
        "feature_highlight_score":feature_highlight_score,
        "overall_score": overall_score
    }

def evaluate_forum_conversation_summary(text, last_message, summary, pii_list=[]):
    # Metrics for RecordTypeForumConversation
    summary = preprocess_text(summary)
    
    coverage_score = keyword_overlap(text, summary)
    repitition_score = detect_repetitive_phrases(summary)
    text_sentences = len(preprocess_text(text).split('.'))
    summary_sentences = len(summary.split('.'))
    summary_to_text_ratio = summary_sentences / text_sentences
    conciseness_threshold = 0.7
    conciseness_score = 1 if summary_to_text_ratio <= conciseness_threshold else 0 
    
    raw_clarity_score = max(0, calculate_flesch_reading_ease(summary)) if summary.lower() != "none" else 0
    clarity_score = raw_clarity_score / 206.835
    last_msg_coverage_score = last_message_coverage(text, last_message, summary)
    context_inclusion_score = context_inclusion(text, last_message, summary)
    pii_removal_score = pii_detection(summary, pii_list)

    # Weights for RecordTypeForumConversation
    weights = {
        "coverage": 0.1,
        "repitition": 0.1,
        "conciseness": 0.1,
        "clarity": 0.05,
        "last_message_coverage": 0.5,
        "context_inclusion": 0.2,
        "pii_removal": 1.0,
    }

    # Calculate the final evaluation score
    overall_score = (
        coverage_score * weights["coverage"]
        + repitition_score * weights["repitition"]
        + conciseness_score * weights["conciseness"]
        #+ clarity_score * weights["clarity"]
        + last_msg_coverage_score * weights["last_message_coverage"]
        + context_inclusion_score * weights["context_inclusion"]
        + pii_removal_score * weights["pii_removal"]
    )
    return {
        "coverage_score": coverage_score,
        #"clarity_score": clarity_score,
        "repitition_score":repitition_score,
        "conciseness_score": conciseness_score,
        "last_message_coverage_score": last_msg_coverage_score,
        "context_inclusion_score":context_inclusion_score,
        "pii_removal_score": pii_removal_score,
        "overall_score": overall_score
    }

def evaluate_survey_summary(product_name, text, summary):
    # Metrics for RecordTypeSurvey
    summary = preprocess_text(summary)
    
    coverage_score = keyword_overlap(text, summary)
    repitition_score = detect_repetitive_phrases(summary)
    text_sentences = len(preprocess_text(text).split('.'))
    summary_sentences = len(summary.split('.'))
    summary_to_text_ratio = summary_sentences / text_sentences
    conciseness_threshold = 0.7
    conciseness_score = 1 if summary_to_text_ratio <= conciseness_threshold else 0     
    
    raw_clarity_score = max(0, calculate_flesch_reading_ease(summary)) if summary.lower() != "none" else 0
    clarity_score = raw_clarity_score / 206.835
    sentiment_align_score = sentiment_alignment(text, summary)
    feature_highlight_score = feature_highlighting(text, summary)

    # Weights for RecordTypeSurvey
    weights = {
        "coverage": 0.1,
        "repitition": 0.1,
        "conciseness": 0.1,
        "clarity": 0.05,
        "sentiment_alignment": 0.2,
        "feature_highlighting": 0.2,
    }

    # Calculate the final evaluation score
    overall_score = (
        coverage_score * weights["coverage"]
        + repitition_score * weights["repitition"]
        + conciseness_score * weights["conciseness"]
        #+ clarity_score * weights["clarity"]
        + sentiment_align_score * weights["sentiment_alignment"]
        + feature_highlight_score * weights["feature_highlighting"]
    )
    return {
        "coverage_score": coverage_score,
        "repitition_score": repitition_score,
        #"clarity_score": clarity_score,
        'conciseness_score':conciseness_score,
        "sentiment_align_score": sentiment_align_score,
        "feature_highlight_score":feature_highlight_score,
        "overall_score": overall_score
    }


def evaluate_summary(record_type, text, summary, product_name=None, pii_list=[]):
    if record_type == "Appstore/Playstore":
        return evaluate_review_summary(product_name, text, summary)
    elif record_type == "Twitter":
        last_message = extract_last_message(text)
        return evaluate_forum_conversation_summary(text, last_message, summary, pii_list)
    elif record_type == "G2":
        return evaluate_survey_summary(product_name, text, summary)
    else:
        raise ValueError(f"Unknown record type: {record_type}")


if __name__=="__main__":
    text = """"User: It makes it so much easier to design for different layouts. I wish there were more breakpoint options as I don't want to use Pugin for that! It would be great to see Figma fully support tokens and modes. 
    User: Plugins! So many custom plugins make my work so easy to do. The product itself is easy to navigate and simple, and there are plenty of tutorials to help when I'm stuck.
    User: The animation part is still quite glitchy. I often find components revert to original when I use animations, which is annoying. It's especially problematic with buttons and on hover animation."
    """
    summary = """ Users appreciate Figma for its prototyping and wireframing capabilities, as well as its handy plugins and UI tools. However, they find the limited options for exporting designs to different formats, including local ones, to be a drawback."""
    product_name = 'figma'
    record_type='G2'
    output = evaluate_summary(record_type, text, summary, product_name)
    output_threshold = {'G2': 0.3, 'Appstore/Playstore': 0.3, 'Twitter': 1.6}
    
    if output["overall_score"] >= output_threshold[record_type]:
        output["label"] = "good"
    else:
        output["label"] = "bad"
    print(output)