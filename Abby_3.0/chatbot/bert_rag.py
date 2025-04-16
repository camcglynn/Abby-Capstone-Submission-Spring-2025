# --- START OF FILE bert_rag.py ---

import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
from utils.data_loader import load_reproductive_health_data
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from chatbot.citation_manager import CitationManager, Citation

# Check for required NLTK resources - main.py should have already downloaded them
# If not, we'll download them here as a fallback
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class BertRAGModel:
    """
    Enhanced BERT-based Retrieval-Augmented Generation model for reproductive health information
    with improved vector search, hybrid retrieval, and reranking.
    (Reverted to version before explicit confidence threshold changes)
    """
    def __init__(self):
        """Initialize the BERT RAG model with improved embeddings and retrieval algorithms"""
        logger.info("Initializing BERT RAG Model")
        try:
            # Load pre-trained model and tokenizer
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # Stemmer for text preprocessing
            self.stemmer = PorterStemmer()

            # Define stopwords manually to avoid NLTK dependency issues
            self.stop_words = {
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                'when', 'while', 'who', 'whom', 'where', 'why', 'how', 'all', 'any', 'both',
                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
                'don', 'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under',
                'again', 'further', 'for', 'of', 'by', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'with',
                'at', 'on', 'by'
            }

            # Load and index the data
            self.qa_pairs = load_reproductive_health_data()

            # Build both vector and keyword indexes
            self.build_indexes()

            # Configure additional retrieval settings
            self.synonyms = self._load_synonyms()

            logger.info("BERT RAG Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BERT RAG Model: {str(e)}", exc_info=True)
            raise

    def _load_synonyms(self):
        """Load reproductive health synonyms for query expansion"""
        # Dictionary of common synonyms and related terms for query expansion
        return {
            "abortion": ["pregnancy termination", "terminate pregnancy", "abortion care"],
            "birth control": ["contraception", "contraceptive", "birth control methods"],
            "std": ["sexually transmitted disease", "sexually transmitted infection", "sti"],
            "sti": ["sexually transmitted infection", "sexually transmitted disease", "std"],
            "morning after pill": ["plan b", "emergency contraception"],
            "iud": ["intrauterine device", "coil"],
            "menstruation": ["period", "menstrual cycle", "monthly bleeding"],
            "pregnancy": ["pregnant", "expecting", "conception"],
            "safe sex": ["protected sex", "safer sex", "condom use"],
            "rape": ["sexual assault", "sexual violence"],
        }

    def build_indexes(self):
        """Build both vector and keyword indexes for hybrid search"""
        logger.info("Building FAISS index for RAG model")
        try:
            # Extract questions and generate embeddings for vector search
            questions = [qa['Question'] for qa in self.qa_pairs]
            self.question_embeddings = self.generate_embeddings(questions)

            # Create FAISS index
            self.dimension = self.question_embeddings.shape[1]
            # Use IndexFlatIP for inner product similarity (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.question_embeddings)

            # Store processed questions for later use
            self.raw_questions = questions
            self.processed_questions = [self._preprocess_text(q) for q in questions]

            logger.info(f"FAISS index built with {len(questions)} questions")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}", exc_info=True)
            raise

    def _preprocess_text(self, text):
        """Preprocess text for keyword search (tokenization, lowercasing, stemming)"""
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)

        # Simple tokenization without relying on nltk's word_tokenize
        tokens = text.split()

        # Remove stopwords and stem
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]

        return tokens

    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using BERT

        Args:
            texts (list): List of text strings to embed

        Returns:
            numpy.ndarray: Normalized embeddings
        """
        embeddings = []

        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize and get BERT embeddings
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use mean pooling to get sentence embeddings
            attention_mask = inputs['attention_mask']
            mean_embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)

            # Normalize
            normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            embeddings.append(normalized_embeddings.numpy())

        return np.vstack(embeddings)

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings

        Args:
            token_embeddings: Token embeddings from BERT
            attention_mask: Attention mask for padding

        Returns:
            torch.Tensor: Mean-pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _is_conversational_query(self, question):
        """Check if the query is conversational rather than informational."""
        question = question.lower()
        greetings = ["hi", "hello", "hey", "greetings", "howdy", "how are you"]
        goodbyes = ["bye", "goodbye", "see you", "farewell", "exit", "quit"]

        # Check if any greeting or goodbye is in the question
        is_greeting = any(greeting in question for greeting in greetings)
        is_goodbye = any(goodbye in question for goodbye in goodbyes)

        if is_goodbye:
            return "goodbye"
        elif is_greeting:
            return "greeting"
        return False

    def _is_out_of_scope(self, question):
        """Check if question is outside the scope of reproductive health"""
        question_lower = question.lower()

        # Special check for ethical questions or general abortion types
        if "abortion murder" in question_lower or "abortion killing" in question_lower or "abortion moral" in question_lower or "abortion ethics" in question_lower or "types of abortion" in question_lower or "different types of abortion" in question_lower or "abortion methods" in question_lower:
            return False  # Not out of scope, use RAG first then optionally ask about policy

        # Define reproductive health related terms
        reproductive_health_terms = ["birth control", "contraception", "pregnancy", "abortion", "std", "sti",
                                   "sexually transmitted", "period", "menstruation", "reproductive", "fertility",
                                   "sex", "sexual", "condom", "iud", "pill", "morning after", "plan b",
                                   "reproductive health", "vagina", "penis", "uterus", "cervix", "sperm",
                                   "egg", "embryo", "fetus", "gestation", "trimester", "ovulation"]

        # Define common out-of-scope topics and their related keywords
        out_of_scope_topics = {
            "weather": ["weather", "temperature", "rain", "sunny", "cloudy", "snow", "forecast", "climate", "humidity", "degrees", "hot", "cold"],
            "politics": ["politics", "election", "government", "president", "vote", "congress", "senate", "political", "democrat", "republican", "law", "politician"],
            "technology": ["computer", "smartphone", "laptop", "tablet", "software", "app", "code", "program", "device", "internet", "website", "tech", "gadget"],
            "travel": ["travel", "flight", "hotel", "vacation", "trip", "tourism", "destination", "airfare", "airline", "beach", "resort", "passport"],
            "food": ["recipe", "cook", "food", "meal", "restaurant", "cuisine", "ingredient", "diet", "nutrition", "eat", "dinner", "lunch", "breakfast"],
            "sports": ["game", "team", "player", "score", "win", "lose", "sport", "match", "tournament", "champion", "football", "basketball", "soccer", "baseball"],
            "entertainment": ["movie", "film", "show", "series", "actor", "music", "song", "artist", "album", "concert", "tv", "television", "celebrity", "theater"],
            "emotional_expression": ["hate", "love", "angry", "mad", "upset", "sad", "depressed", "lonely", "happy", "excited", "worried", "anxious", "scared", "afraid", "hurt", "pain", "suffer", "confused", "lost", "hopeless", "helpless", "tired", "exhausted", "overwhelmed", "stressed"]
        }

        # Check if any reproductive health terms are in the question
        has_reproductive_terms = any(term in question_lower for term in reproductive_health_terms)

        # Check if any out of scope topics are in the question
        detected_topics = []
        for topic, keywords in out_of_scope_topics.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_topics.append(topic)

        # Check for general greeting or off-topic patterns
        general_query = self._is_general_query(question_lower)

        # If no reproductive health terms are present and we have detected out-of-scope topics,
        # or if the query appears to be about something else entirely
        if (not has_reproductive_terms and detected_topics) or general_query:
            return detected_topics if detected_topics else ["general"]

        return False

    def _is_general_query(self, question_lower):
        """Check if this is a general query unrelated to reproductive health"""
        # General questions and off-topic patterns
        general_patterns = [
            "what's the time", "what time is it", "what day is it", "what is your name",
            "who are you", "what can you do", "how old are you", "where are you from",
            "tell me a joke", "tell me about yourself", "what's your favorite", "what is your favorite",
            "hello world", "test", "can you help me with", "what's happening", "what is happening",
            "how's it going", "how does this work", "what's new", "what is new"
        ]

        # Check if question matches any general patterns
        for pattern in general_patterns:
            if pattern in question_lower:
                return True

        # Check if question is very short (likely not reproductive health specific)
        # and doesn't contain any health-related terms
        health_terms = ["health", "medical", "doctor", "pill", "pregnancy", "sex", "birth"]
        if len(question_lower.split()) <= 3 and not any(term in question_lower for term in health_terms):
            return True

        return False

    def expand_query(self, question):
        """
        Expand query with synonyms and related terms for better recall

        Args:
            question (str): The original user question

        Returns:
            str: Expanded question with relevant terms
        """
        question_lower = question.lower()
        expansions = []

        # Add synonyms for terms in the question
        for term, synonyms in self.synonyms.items():
            if term in question_lower:
                for synonym in synonyms:
                    # Only add if not already in the question
                    if synonym not in question_lower:
                        expansions.append(synonym)

        # Add expansions if found
        if expansions:
            expanded_question = f"{question} {' '.join(expansions)}"
            logger.debug(f"Expanded query: '{question}' -> '{expanded_question}'")
            return expanded_question

        return question

    def get_semantic_similarity(self, text1, text2):
        """
        Calculate semantic similarity between two texts

        Args:
            text1 (str): First text
            text2 (str): Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        # Generate embeddings
        embeddings = self.generate_embeddings([text1, text2])

        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1])

        return float(similarity)

    def get_response(self, question, top_k=5):
        """
        Get response for a given question using RAG

        Args:
            question (str): The question to answer
            top_k (int): Number of similar questions to retrieve

        Returns:
            str: The answer to the question
        """
        try:
            # Import citation manager
            citation_mgr = CitationManager()

            # Check if this is a conversational query instead of a health question
            conversational_type = self._is_conversational_query(question)
            if conversational_type == "greeting":
                logger.debug(f"Detected conversational query: '{question}'")
                greeting_response = "I'm doing well, thanks for asking! How can I help you with reproductive health information today?"
                # Ensure Planned Parenthood source exists before adding citation
                if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                return citation_mgr.add_citation_to_text(greeting_response, "planned_parenthood")
            elif conversational_type == "goodbye":
                goodbye_response = "Goodbye! Take care and stay healthy."
                if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                return citation_mgr.add_citation_to_text(goodbye_response, "planned_parenthood")


            # Check if the question is out of scope
            out_of_scope = self._is_out_of_scope(question)
            if out_of_scope:
                topics = ", ".join(out_of_scope)
                logger.debug(f"Detected out-of-scope query about {topics}: '{question}'")
                out_of_scope_response = self._get_out_of_scope_response(out_of_scope)
                if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                # Don't add citation if it was an emotional expression
                if "NO_CITATION" in out_of_scope_response:
                     return out_of_scope_response.replace("NO_CITATION", "")
                else:
                     return citation_mgr.add_citation_to_text(out_of_scope_response, "planned_parenthood")

            # First check for exact matches (case-insensitive) to prioritize them
            normalized_question = question.lower().strip('?. ')

            # Check for exact match
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                # Check if this is an exact match
                if normalized_question == qa_normalized:
                    logger.debug(f"Found exact match for question: '{question}'")
                    logger.debug(f"Exact match index: {idx}")
                    answer = qa_pair['Answer']
                    # Use the link from the QA pair if available
                    link = qa_pair.get('URL', qa_pair.get('Link', '')) # Check both URL and Link
                    if link and isinstance(link, str) and len(link) > 0:
                        # Create a custom citation for this specific link
                        source_name = qa_pair.get('Source', 'Planned Parenthood') # Use provided source or default
                        citation_id = f"custom_{idx}"
                        citation_mgr.sources[citation_id] = Citation(
                            source=source_name,
                            url=link,
                            title=qa_pair['Question'], # Use question as title
                            authors=[] # Assume no specific authors unless data provides them
                        )
                        return citation_mgr.add_citation_to_text(answer, citation_id)
                    else:
                        # Add default PP citation if no link
                        if "planned_parenthood" not in citation_mgr.sources:
                           citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                        return citation_mgr.add_citation_to_text(answer, "planned_parenthood")


            # Also check for questions that contain the exact query
            # This helps with cases like "what is the menstrual cycle" matching "what is the menstrual cycle?"
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                if qa_normalized.startswith(normalized_question) or normalized_question.startswith(qa_normalized):
                    logger.debug(f"Found partial match for question: '{question}'")
                    logger.debug(f"Partial match index: {idx}")
                    answer = qa_pair['Answer']
                    # Use the link from the QA pair if available
                    link = qa_pair.get('URL', qa_pair.get('Link', ''))
                    if link and isinstance(link, str) and len(link) > 0:
                        # Create a custom citation for this specific link
                        source_name = qa_pair.get('Source', 'Planned Parenthood')
                        citation_id = f"custom_{idx}"
                        citation_mgr.sources[citation_id] = Citation(
                            source=source_name,
                            url=link,
                            title=qa_pair['Question'],
                            authors=[]
                        )
                        return citation_mgr.add_citation_to_text(answer, citation_id)
                    else:
                        if "planned_parenthood" not in citation_mgr.sources:
                           citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                        return citation_mgr.add_citation_to_text(answer, "planned_parenthood")

            # Expand the query for better recall
            expanded_question = self.expand_query(question)

            # If no exact match, proceed with embedding-based retrieval
            # Generate embedding for the question
            question_embedding = self.generate_embeddings([expanded_question])

            # Check if embeddings were generated
            if question_embedding is None or question_embedding.size == 0:
                 logger.error("Failed to generate embedding for the query.")
                 return citation_mgr.add_citation_to_text("I'm having trouble processing your question right now. Please try again.", "planned_parenthood")


            # Search for similar questions
            # Ensure k is not larger than the number of items in the index
            k = min(top_k, self.index.ntotal)
            if k <= 0:
                 logger.warning("FAISS index is empty or k=0, cannot search.")
                 return citation_mgr.add_citation_to_text("I don't have information matching that query right now.", "planned_parenthood")

            distances, indices = self.index.search(question_embedding.astype(np.float32), k)

            # Original confidence check was here - removed as per request to revert threshold changes
            # if distances[0][0] > 15.0: ...

            # If multiple good matches found (original check was < 12.0 distance for second match)
            # Reverting this combination logic as well to match the "original" state requested
            # if len(indices[0]) > 1 and distances[0][1] < 12.0:
            #    return self._combine_top_answers(question, distances[0], indices[0])

            # Get the most similar question's answer (index 0)
            if indices.size > 0 and indices[0].size > 0:
                 best_idx = indices[0][0]
                 # Check if best_idx is valid
                 if 0 <= best_idx < len(self.qa_pairs):
                      best_answer = self.qa_pairs[best_idx]['Answer']
                      best_question = self.qa_pairs[best_idx]['Question']

                      # Add citation with the link from the QA pair if available
                      link = self.qa_pairs[best_idx].get('URL', self.qa_pairs[best_idx].get('Link', ''))
                      if link and isinstance(link, str) and len(link) > 0:
                           # Create a custom citation for this specific link
                           source_name = self.qa_pairs[best_idx].get('Source', 'Planned Parenthood')
                           citation_id = f"custom_{best_idx}"
                           citation_mgr.sources[citation_id] = Citation(
                                source=source_name,
                                url=link,
                                title=best_question,
                                authors=[]
                           )
                           cited_answer = citation_mgr.add_citation_to_text(best_answer, citation_id)
                      else:
                           if "planned_parenthood" not in citation_mgr.sources:
                              citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                           cited_answer = citation_mgr.add_citation_to_text(best_answer, "planned_parenthood")

                      # Log basic info without the specific thresholds
                      logger.debug(f"Matched question: {best_question}")

                      # Return the answer with citation
                      return cited_answer
                 else:
                      logger.error(f"FAISS returned invalid index: {best_idx}")
                      return citation_mgr.add_citation_to_text("I found related information, but had trouble retrieving the specific answer. Could you try rephrasing?", "planned_parenthood")
            else:
                logger.warning(f"FAISS search returned no results for query: '{question}'")
                return citation_mgr.add_citation_to_text("I couldn't find specific information matching your question. Could you try asking differently?", "planned_parenthood")


        except Exception as e:
            logger.error(f"Error getting RAG response: {str(e)}", exc_info=True)
            error_response = "I apologize, but I encountered an error processing your question. Please try asking again or rephrase your question."

            # Add citation even for error responses
            try:
                if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                return citation_mgr.add_citation_to_text(error_response, "planned_parenthood")
            except:
                # If citation fails, return plain error message
                return error_response

    def get_response_with_context(self, question, top_k=5):
        """
        Get response with context information for a given question using RAG
        (Reverted to version before explicit confidence threshold changes)

        Args:
            question (str): The question to answer
            top_k (int): Number of similar questions to retrieve

        Returns:
            tuple: (answer string, context list)
        """
        # This implementation mirrors get_response but also returns the contexts.
        # Reverting confidence checks here too.
        try:
            citation_mgr = CitationManager()
            contexts = []

            conversational_type = self._is_conversational_query(question)
            if conversational_type == "greeting":
                greeting_response = "I'm doing well, thanks for asking! How can I help you with reproductive health information today?"
                if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                return citation_mgr.add_citation_to_text(greeting_response, "planned_parenthood"), contexts
            elif conversational_type == "goodbye":
                goodbye_response = "Goodbye! Take care and stay healthy."
                if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                return citation_mgr.add_citation_to_text(goodbye_response, "planned_parenthood"), contexts

            out_of_scope = self._is_out_of_scope(question)
            if out_of_scope:
                topics = ", ".join(out_of_scope)
                out_of_scope_response = self._get_out_of_scope_response(out_of_scope)
                if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                if "NO_CITATION" in out_of_scope_response:
                    return out_of_scope_response.replace("NO_CITATION", ""), contexts
                else:
                    return citation_mgr.add_citation_to_text(out_of_scope_response, "planned_parenthood"), contexts


            normalized_question = question.lower().strip('?. ')
            # Check for exact match
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                if normalized_question == qa_normalized:
                    answer = qa_pair['Answer']
                    link = qa_pair.get('URL', qa_pair.get('Link', ''))
                    contexts.append({'question': qa_pair['Question'], 'answer': answer, 'distance': 0.0, 'category': qa_pair.get('Category', 'General'), 'source': qa_pair.get('Source', 'Planned Parenthood'), 'link': link})
                    if link and isinstance(link, str) and len(link) > 0:
                        source_name = qa_pair.get('Source', 'Planned Parenthood')
                        citation_id = f"custom_{idx}"
                        citation_mgr.sources[citation_id] = Citation(source=source_name, url=link, title=qa_pair['Question'], authors=[])
                        return citation_mgr.add_citation_to_text(answer, citation_id), contexts
                    else:
                         if "planned_parenthood" not in citation_mgr.sources:
                             citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                         return citation_mgr.add_citation_to_text(answer, "planned_parenthood"), contexts

            # Check for partial match
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                if qa_normalized.startswith(normalized_question) or normalized_question.startswith(qa_normalized):
                    answer = qa_pair['Answer']
                    link = qa_pair.get('URL', qa_pair.get('Link', ''))
                    contexts.append({'question': qa_pair['Question'], 'answer': answer, 'distance': 1.0, 'category': qa_pair.get('Category', 'General'), 'source': qa_pair.get('Source', 'Planned Parenthood'), 'link': link})
                    if link and isinstance(link, str) and len(link) > 0:
                        source_name = qa_pair.get('Source', 'Planned Parenthood')
                        citation_id = f"custom_{idx}"
                        citation_mgr.sources[citation_id] = Citation(source=source_name, url=link, title=qa_pair['Question'], authors=[])
                        return citation_mgr.add_citation_to_text(answer, citation_id), contexts
                    else:
                         if "planned_parenthood" not in citation_mgr.sources:
                             citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                         return citation_mgr.add_citation_to_text(answer, "planned_parenthood"), contexts

            expanded_question = self.expand_query(question)
            question_embedding = self.generate_embeddings([expanded_question])
            if question_embedding is None or question_embedding.size == 0:
                 logger.error("Failed to generate embedding for the query (with context).")
                 return citation_mgr.add_citation_to_text("I'm having trouble processing your question right now. Please try again.", "planned_parenthood"), []


            k = min(top_k, self.index.ntotal)
            if k <= 0:
                 logger.warning("FAISS index is empty or k=0, cannot search (with context).")
                 return citation_mgr.add_citation_to_text("I don't have information matching that query right now.", "planned_parenthood"), []

            distances, indices = self.index.search(question_embedding.astype(np.float32), k)

            # Populate contexts list regardless of confidence threshold
            retrieved_scores = []
            if indices.size > 0:
                 retrieved_scores = distances[0].tolist()
                 for i in range(min(k, len(indices[0]))):
                     idx = indices[0][i]
                     dist = float(distances[0][i])
                     if 0 <= idx < len(self.qa_pairs):
                          row = self.qa_pairs[idx]
                          contexts.append({
                               'question': row['Question'],
                               'answer': row['Answer'],
                               'distance': dist,
                               'category': row.get('Category', 'General'),
                               'source': row.get('Source', 'Planned Parenthood'),
                               'link': row.get('URL', row.get('Link', ''))
                          })
                     else:
                          logger.warning(f"Invalid index {idx} from FAISS search.")


            # Reverting confidence check: Immediately proceed to get best answer
            if indices.size > 0 and indices[0].size > 0:
                 best_idx = indices[0][0]
                 if 0 <= best_idx < len(self.qa_pairs):
                      best_answer = self.qa_pairs[best_idx]['Answer']
                      best_question = self.qa_pairs[best_idx]['Question']
                      link = self.qa_pairs[best_idx].get('URL', self.qa_pairs[best_idx].get('Link', ''))

                      if link and isinstance(link, str) and len(link) > 0:
                           source_name = self.qa_pairs[best_idx].get('Source', 'Planned Parenthood')
                           citation_id = f"custom_{best_idx}"
                           citation_mgr.sources[citation_id] = Citation(source=source_name, url=link, title=best_question, authors=[])
                           cited_answer = citation_mgr.add_citation_to_text(best_answer, citation_id)
                      else:
                           if "planned_parenthood" not in citation_mgr.sources:
                               citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                           cited_answer = citation_mgr.add_citation_to_text(best_answer, "planned_parenthood")

                      logger.debug(f"Matched question (with context): {best_question}")
                      return cited_answer, contexts
                 else:
                      logger.error(f"FAISS returned invalid index {best_idx} (with context).")
                      return citation_mgr.add_citation_to_text("I found related information, but had trouble retrieving the specific answer. Could you try rephrasing?", "planned_parenthood"), contexts
            else:
                logger.warning(f"FAISS search returned no results for query (with context): '{question}'")
                return citation_mgr.add_citation_to_text("I couldn't find specific information matching your question. Could you try asking differently?", "planned_parenthood"), contexts


        except Exception as e:
            logger.error(f"Error getting RAG response with context: {str(e)}", exc_info=True)
            error_response = "I apologize, but I encountered an error processing your question. Please try asking again or rephrase your question."
            try:
                 if "planned_parenthood" not in citation_mgr.sources:
                    citation_mgr.sources["planned_parenthood"] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")
                 return citation_mgr.add_citation_to_text(error_response, "planned_parenthood"), []
            except:
                return error_response, []

    def _combine_top_answers(self, question, distances, indices, max_answers=3):
        """
        Combine the top answers for better response with natural language flow
        (Reverted to version before major formatting changes)

        Args:
            question (str): The original question
            distances (list): Distances of top matches
            indices (list): Indices of top matches
            max_answers (int): Maximum number of answers to combine

        Returns:
            str: Combined answer
        """
        relevant_answers = []
        citation_mgr = CitationManager() # Instance for this combination

        # Get the top answers
        for i in range(min(max_answers, len(indices))):
            # Original code didn't have a strict distance check here, just combined top K if second was close enough
            idx = indices[i]
            if 0 <= idx < len(self.qa_pairs):
                answer = self.qa_pairs[idx]['Answer']
                link = self.qa_pairs[idx].get('URL', self.qa_pairs[idx].get('Link', ''))
                source = self.qa_pairs[idx].get('Source', 'Planned Parenthood')
                title = self.qa_pairs[idx]['Question']
                citation_id = f"combined_{i}_{idx}"

                # Add citation source
                if link and isinstance(link, str) and len(link) > 0:
                    citation_mgr.sources[citation_id] = Citation(source=source, url=link, title=title, authors=[])
                else:
                    # Use default PP source if no link
                    citation_id = "planned_parenthood"
                    if citation_id not in citation_mgr.sources:
                        citation_mgr.sources[citation_id] = Citation(source="Planned Parenthood", url="https://www.plannedparenthood.org/")

                # Add citation marker to the answer chunk
                cited_answer_chunk = citation_mgr.add_citation_to_text(answer, citation_id)
                relevant_answers.append(cited_answer_chunk)

        # Simple combination: join the answers (potentially with citations)
        combined_response = " ".join(relevant_answers)

        # Basic cleanup (remove duplicate citations if they were added multiple times)
        final_response = re.sub(r'(\[\^[\w_]+\]\s*)+', r'\1', combined_response) # Keep only last marker if repeated

        return final_response

    def _format_single_answer(self, question, answer):
        """
        Format a single answer to be concise and direct (original simple version)

        Args:
            question (str): Original question
            answer (str): Answer to format

        Returns:
            str: Formatted answer
        """
        # Keep the original answer as is, maybe add punctuation if missing
        if not answer.endswith(('.', '?', '!')):
            answer = answer + '.'
        return answer


    def _format_multiple_answers(self, question, relevant_answers):
        """
        Format multiple answers into a cohesive, concise response (original simple version)

        Args:
            question (str): Original question
            relevant_answers (list): List of relevant answer dictionaries (already cited text)

        Returns:
            str: Formatted and combined answer
        """
        # Simple join, relying on _combine_top_answers to have added citations already
        combined_response = " ".join([item['answer'] for item in relevant_answers]) # Assume item is dict with 'answer' key containing cited text

        # Basic cleanup (remove duplicate citations if they were added multiple times)
        final_response = re.sub(r'(\[\^[\w_]+\]\s*)+', r'\1', combined_response)

        return final_response

    def _get_first_sentences(self, text, num_sentences=2):
        """Extract the first N sentences from a text"""
        import re
        # Improved sentence splitting regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty strings that might result from splitting
        sentences = [s for s in sentences if s]
        return ' '.join(sentences[:num_sentences])


    def _extract_sentences(self, text):
        """Split text into sentences"""
        import re
        # Improved sentence splitting regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty strings
        return [s for s in sentences if s]


    def _get_key_phrases(self, text):
        """Extract key phrases/words from text to identify content"""
        # Simple implementation: just use non-stop words of 4+ chars
        words = [w.lower() for w in text.split() if len(w) >= 4 and w.lower() not in self.stop_words]
        return set(words)

    # Removed is_confident, _is_semantically_similar, _get_semantic_similarity_score
    # as they were added after the "original" state being requested.

    def _get_out_of_scope_response(self, topics):
        """Generate a response for out-of-scope questions"""
        # Default response for out-of-scope topics
        primary_topic = topics[0] if topics else "general"

        topic_responses = {
            "weather": "I'm designed to provide information about reproductive health, not weather forecasts. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "politics": "I'm here to provide information about reproductive health, not political matters. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "technology": "I'm programmed to assist with reproductive health questions, not technology matters. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "travel": "I'm designed to answer questions about reproductive health, not travel information. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "food": "I'm trained to provide information about reproductive health, not food or nutrition in general. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "sports": "I'm programmed to assist with reproductive health questions, not sports information. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "entertainment": "I'm designed to provide information about reproductive health, not entertainment. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those instead.",
            "emotional_expression": "I understand you're expressing a personal feeling. While I'm here to listen, I'm specifically designed to provide information about reproductive health topics. If you have any questions about contraception, pregnancy, or sexual health, I'd be happy to help with those.",
            "general": "I'm Abby, a chatbot specifically designed to provide information about reproductive health. I don't have information on this topic. If you have any questions about contraception, pregnancy, abortion access, or sexual health, I'd be happy to help with those instead."
        }

        # Check if this is emotional expression - if so, we'll return the response WITHOUT citation
        if primary_topic == "emotional_expression":
            # Return without citation by setting a special flag that the caller should check
            return topic_responses["emotional_expression"] + "NO_CITATION"

        return topic_responses.get(primary_topic, topic_responses["general"])

# --- END OF FILE bert_rag.py ---