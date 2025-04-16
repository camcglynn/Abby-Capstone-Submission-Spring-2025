# --- START OF FILE knowledge_handler.py ---

import logging
import os
import asyncio
# import aiohttp # Removed as not used directly here
import numpy as np
import pandas as pd
import re
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from datetime import datetime
import json
# Use AsyncOpenAI if available for async processing
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
    from openai import OpenAI # Fallback to sync OpenAI

from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class KnowledgeHandler:
    """
    Handler for knowledge-seeking aspects of user queries.

    This class processes factual questions about reproductive health,
    accessing reliable knowledge sources (RAG) and generating informative responses using OpenAI.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the knowledge handler

        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            model_name (str): OpenAI model to use
        """
        logger.info(f"Initializing KnowledgeHandler with model {model_name}")

        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
             # Allow initialization without API key, RAG will still work
             logger.warning("OpenAI API key not found. Knowledge handler will rely solely on RAG.")
             self.client = None
             self.is_async = False
        else:
            # Set up OpenAI client (try async first, fallback to sync)
            if AsyncOpenAI:
                 self.client = AsyncOpenAI(api_key=self.api_key)
                 self.is_async = True
            else:
                 self.client = OpenAI(api_key=self.api_key)
                 self.is_async = False
                 logger.warning("AsyncOpenAI not found, using synchronous OpenAI client for knowledge generation.")

        self.model = model_name

        # Initialize data structures
        self.data = pd.DataFrame()
        self.index = None
        self.embeddings = None

        # Try to load BERT model for embeddings
        try:
            self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            logger.info(f"Successfully loaded embedding model: {self.embedding_model_name}")

            # Load datasets and build index
            self.data = self._load_datasets()
            if not self.data.empty:
                 self.index, self.embeddings = self._build_index()
            else:
                 logger.error("Failed to load datasets. RAG functionality will be disabled.")

        except Exception as e:
            logger.error(f"Error loading embedding model or building index: {str(e)}")
            self.embedding_model = None
            self.tokenizer = None
            self.index = None
            self.embeddings = None


        # --- FIX (Bug 3): Updated Knowledge Prompt for Structured Output ---
        self.knowledge_prompt_template = """
You are Abby, a knowledgeable and compassionate reproductive health specialist providing factual information in an empathetic tone.
User query: {query}
Using the following knowledge sources to inform your response:
{knowledge_sources}

--- OUTPUT STRUCTURE REQUIREMENTS ---
Generate the response adhering STRICTLY to the following structure and guidelines:

1.  **Introduction:** Start with a SINGLE introductory sentence that acknowledges the query's topic sensitively and summarizes the main point briefly.
2.  **Section Delimiter:** Follow the introduction with a blank line.
3.  **Section Content & Tone:** For each distinct point or aspect covered in the knowledge sources relevant to the query:
    *   Start a new section with a delimiter `###TITLE_TEXT###` where TITLE_TEXT is a concise, descriptive title for the section (e.g., `###What is Abortion###`, `###Types of Procedures###`). The delimiter MUST be on its own line.
    *   Follow the delimiter line IMMEDIATELY with the paragraph(s) explaining that section's topic.
    *   **CONTENT SOURCE:** Use information **ONLY** from the provided knowledge sources. Do not add outside information or opinions.
    *   **TONE (CRITICAL):** Balance **empathy** with **factual accuracy**. Use clear, sensitive, supportive, and non-judgmental language. Avoid overly clinical or alarming terms where possible, but prioritize medical accuracy. Ensure the tone is helpful and understanding.
    *   **MEDICAL TERMS:** Use **bold formatting** for key medical terms when first introduced within the explanation text itself (NOT in the title delimiter). Use markdown `**term**`.
    *   **SEPARATION:** Separate each `###TITLE###` block (delimiter + text) from the next with a SINGLE blank line.
4.  **EXCLUSIONS:** **DO NOT** add your own "Sources:", "References:", "Citations:", "Disclaimer:", or any similar list/section at the end of your response. Citation information is handled separately. Do not add conversational filler like "I hope this helps".

--- EXAMPLE OF REQUIRED OUTPUT FORMAT ---
Birth control methods offer various ways to prevent pregnancy, fitting different needs and lifestyles.

###Types of Birth Control###
There are many types available. **Hormonal methods** like the **pill**, **patch**, **ring**, **shot**, and **implant** use hormones to prevent ovulation. **Barrier methods** like **condoms** and **diaphragms** block sperm. **Intrauterine devices (IUDs)**, both hormonal and copper, are placed in the uterus for long-term prevention. **Sterilization** offers a permanent option.

###How Birth Control Works###
Most methods work by preventing sperm from reaching an egg (**fertilization**) or stopping the release of an egg (**ovulation**). For example, hormonal methods often stop ovulation, while IUDs can interfere with sperm movement and fertilization. Barrier methods physically block sperm.
--- END OF EXAMPLE ---

Generate the response now using ONLY the provided sources and adhering STRICTLY to the delimiter-based structure (`###TITLE###\nContent\n\n`), the tone requirements, and the instruction to NOT add a sources list or disclaimers.
"""
        # --- End Bug 3 Fix ---


    async def process_query(self, query: str, full_message: str = None, conversation_history: List[Dict[str, Any]] = None, user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        start_time = time.time()
        full_message = full_message or query # Use query if full_message is None
        logger.info(f"Processing knowledge query: {query[:100]}...")

        # Default response structure
        response_structure = {
            "text": "I'm sorry, I couldn't find specific information about that topic. Could you please rephrase your question?",
            "primary_content": "I'm sorry, I couldn't find specific information about that topic. Could you please rephrase your question?",
            "citations": [],
            "citation_objects": [],
            "processing_time": 0.0,
            "aspect_type": "knowledge",
            "question_answered": False,
            "needs_state_info": False
        }

        # Check if RAG is functional
        if self.index is None or self.data.empty:
             logger.error("RAG index or data is not available. Cannot process knowledge query.")
             response_structure["text"] = "I apologize, my knowledge base is currently unavailable. I cannot answer factual questions at this time."
             response_structure["primary_content"] = response_structure["text"]
             response_structure["processing_time"] = time.time() - start_time
             return response_structure

        try:
            # Clean the query text
            query = query.strip()
            if not query:
                response_structure["processing_time"] = time.time() - start_time
                return response_structure

            # Retrieve relevant documents using RAG
            docs, scores = await self._retrieve_context(query, top_k=3) # Retrieve top 3 docs

            # Filter docs based on relevance threshold (for normalized embeddings L2 distance < 1.4 is reasonable)
            relevance_threshold = 1.4 # Updated for normalized embeddings (theoretical max is sqrt(2) ≈ 1.414)
            relevant_docs = [doc for i, doc in enumerate(docs) if scores[i] < relevance_threshold]
            logger.info(f"Retrieved {len(docs)} docs, {len(relevant_docs)} are relevant (threshold: {relevance_threshold}).")

            generated_from_rag = False # Flag to track if RAG was used

            if relevant_docs and self.client:
                # --- FIX (Bug 3): Use updated prompt ---
                formatted_sources = self._format_vector_sources(relevant_docs) # Format sources for prompt
                prompt = self.knowledge_prompt_template.format(
                    query=query,
                    knowledge_sources=formatted_sources
                )
                # --- End Bug 3 Fix ---

                # Generate response using OpenAI
                logger.info(f"Generating knowledge response with OpenAI using {len(relevant_docs)} retrieved documents.")
                if self.is_async:
                    completion = await self.client.chat.completions.create(
                         model=self.model,
                         messages=[{"role": "user", "content": prompt}],
                         temperature=0.5, # Slightly lower temp for factual accuracy
                         max_tokens=700 # Allow slightly longer responses for knowledge
                    )
                else:
                    # Sync fallback
                    completion = self.client.chat.completions.create(
                         model=self.model,
                         messages=[{"role": "user", "content": prompt}],
                         temperature=0.5,
                         max_tokens=700
                    )

                raw_response_text = completion.choices[0].message.content.strip()

                # Basic cleaning (remove potential redundant sign-offs or extra markers)
                cleaned_text = re.sub(r'\n\s*(Sources:|Citations:|References:|Disclaimer:|Please note|I hope this helps|Remember support is available).*', '', raw_response_text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE).strip()
                 # Remove potential duplicate section markers if LLM adds them incorrectly
                cleaned_text = re.sub(r'(###.*?###\s*\n)\1+', r'\1', cleaned_text)

                response_structure["text"] = cleaned_text
                response_structure["primary_content"] = cleaned_text
                response_structure["question_answered"] = True
                generated_from_rag = True # Mark that RAG docs were used for generation

            elif relevant_docs: # Fallback if OpenAI client not available but RAG found docs
                logger.warning("OpenAI client not available, combining RAG results directly.")
                combined_text = self._combine_retrieved_docs(relevant_docs, scores[:len(relevant_docs)])
                # Wrap combined RAG text in paragraph tags for consistency
                formatted_combined_text = f"<p class='message-paragraph'>{combined_text.replace(chr(10), '<br>')}</p>"
                response_structure["text"] = formatted_combined_text
                response_structure["primary_content"] = formatted_combined_text
                response_structure["question_answered"] = True
                generated_from_rag = True # Mark that RAG docs were used for generation

            else: # Fallback if no relevant docs were found by RAG
                logger.warning("No relevant documents found by RAG for the query.")
                # Use the default "not found" message already in response_structure
                response_structure["question_answered"] = False
                generated_from_rag = False # Ensure flag is false if no docs found

            # Populate citations ONLY if RAG docs were actually used for the response
            if generated_from_rag and relevant_docs:
                citation_objects = self._extract_citations(relevant_docs)
                citation_sources = list(set(c["source"] for c in citation_objects))
                response_structure["citations"] = citation_sources
                response_structure["citation_objects"] = citation_objects
                logger.info(f"Added {len(citation_objects)} citations based on used RAG documents.")
            else:
                # Ensure citations are empty if RAG wasn't used or found no docs
                response_structure["citations"] = []
                response_structure["citation_objects"] = []
                logger.info("No citations added as RAG context was not used or unavailable.")

            response_structure["processing_time"] = time.time() - start_time
            logger.info(f"Knowledge response generated in {response_structure['processing_time']:.2f} seconds")

            return response_structure

        except Exception as e:
            logger.error(f"Error in knowledge handler: {str(e)}", exc_info=True)
            response_structure["text"] = "I'm having trouble finding specific information about that right now. Could you try rephrasing your question?"
            response_structure["primary_content"] = response_structure["text"]
            response_structure["processing_time"] = time.time() - start_time
            # Ensure citations are empty on error
            response_structure["citations"] = []
            response_structure["citation_objects"] = []
            response_structure["question_answered"] = False
            return response_structure


    def _find_relevant_sources(self, query: str) -> List[Dict[str, Any]]:
        """Deprecated: Use _retrieve_context instead."""
        logger.warning("_find_relevant_sources is deprecated. Use _retrieve_context.")
        # Simulate async behavior if needed, though it's deprecated
        # await asyncio.sleep(0)
        return [] # Return empty list

    def _format_knowledge_sources(self, sources: List[Dict[str,Any]]) -> str:
        """Deprecated: Use _format_vector_sources instead."""
        logger.warning("_format_knowledge_sources is deprecated. Use _format_vector_sources.")
        return self._format_vector_sources(sources)


    def _format_vector_sources(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format vector search results for inclusion in the prompt.
        Includes URL if available.
        """
        if not docs:
            return "No relevant knowledge sources found."

        formatted_text = ""
        for i, doc in enumerate(docs):
             source_name = doc.get('source', 'Unknown Source')
             url = doc.get('url', '')
             # Create a display name for the source, including URL if present and valid
             source_display = source_name
             if url and isinstance(url, str) and url.startswith(('http://', 'https://')):
                  source_display += f" ({url})"

             formatted_text += f"SOURCE [{i+1}]: {source_display}\n"
             # Optional: Include question context if helpful
             # context_question = doc.get('question', '')
             # if context_question: formatted_text += f"Context Question: {context_question}\n"
             formatted_text += f"Content: {doc.get('answer', '')}\n\n" # Changed label to Content

        return formatted_text.strip()


    def _load_datasets(self) -> pd.DataFrame:
        """
        Load and prepare knowledge datasets from CSV files.

        Returns:
            pd.DataFrame: Combined and cleaned dataset.
        """
        try:
            all_dfs = []
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            
            # Handle AbortionPPDFAQ.csv - First attempt using a more direct approach
            abortion_csv_path = os.path.join(project_root, "data/AbortionPPDFAQ.csv")
            if not os.path.exists(abortion_csv_path):
                abortion_csv_path = os.path.join(os.path.dirname(__file__), "..", "data/AbortionPPDFAQ.csv")
            
            if os.path.exists(abortion_csv_path):
                logger.info(f"Loading AbortionPPDFAQ.csv from {abortion_csv_path}")
                
                # First manually read and skip the header line
                with open(abortion_csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Skip the first line that just says "AbortionPPDFAQ" and create proper content
                if len(lines) > 1:
                    temp_csv_path = os.path.join(os.path.dirname(abortion_csv_path), "_temp_abortion.csv")
                    with open(temp_csv_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines[1:])  # Skip first line
                    
                    # Now read with pandas
                    df_abortion = pd.read_csv(temp_csv_path)
                    
                    # Clean up the temporary file
                    try:
                        os.remove(temp_csv_path)
                    except:
                        pass
                    
                    # Rename columns to standard names
                    column_map = {"question": "Question", "answer": "Answer", "link": "Link"}
                    df_abortion = df_abortion.rename(columns={k: v for k, v in column_map.items() if k in df_abortion.columns})
                    
                    # Print column info for debugging
                    logger.info(f"AbortionPPDFAQ.csv columns after loading: {df_abortion.columns.tolist()}")
                    
                    # Debug: check for Link column
                    if "Link" in df_abortion.columns:
                        sample_links = df_abortion["Link"].head(5).tolist()
                        logger.info(f"AbortionPPDFAQ.csv sample Link values: {sample_links}")
                    
                    # Ensure all required columns exist
                    if "Question" in df_abortion.columns and "Answer" in df_abortion.columns:
                        # Add source if missing
                        if "Source" not in df_abortion.columns:
                            df_abortion["Source"] = "Planned Parenthood Abortion FAQ"
                            
                        # Keep Link column as is - don't rename to URL yet
                        
                        # Keep only necessary columns in standard format
                        standard_columns = ["Question", "Answer", "Source", "Link"]
                        existing_columns = [col for col in standard_columns if col in df_abortion.columns]
                        df_abortion = df_abortion[existing_columns]
                        
                        # Clean data
                        df_abortion = df_abortion.dropna(subset=["Question", "Answer"])
                        
                        # Add to collection
                        all_dfs.append(df_abortion)
                        logger.info(f"Successfully loaded AbortionPPDFAQ.csv with {len(df_abortion)} rows")
                    else:
                        logger.warning(f"AbortionPPDFAQ.csv is missing required columns. Found: {df_abortion.columns.tolist()}")
            else:
                logger.warning("AbortionPPDFAQ.csv not found in either expected location")
            
            # Handle Planned Parenthood Data CSV - This one has proper headers already
            pp_csv_path = os.path.join(project_root, "data/Planned Parenthood Data - Sahana.csv")
            if not os.path.exists(pp_csv_path):
                pp_csv_path = os.path.join(os.path.dirname(__file__), "..", "data/Planned Parenthood Data - Sahana.csv")
            
            if os.path.exists(pp_csv_path):
                logger.info(f"Loading Planned Parenthood Data from {pp_csv_path}")
                
                # This file has proper headers already, just load directly
                df_pp = pd.read_csv(pp_csv_path)
                
                # Print column info for debugging
                logger.info(f"Planned Parenthood Data columns: {df_pp.columns.tolist()}")
                
                # Debug: check for Link column and its values
                if "Link" in df_pp.columns:
                    # Check first 5 link values
                    sample_links = df_pp["Link"].head(5).tolist()
                    logger.info(f"Sample Link values: {sample_links}")
                else:
                    logger.warning("Link column is missing in Planned Parenthood Data")
                
                # Ensure all required columns exist
                if "Question" in df_pp.columns and "Answer" in df_pp.columns:
                    # Add source if missing
                    if "Source" not in df_pp.columns:
                        df_pp["Source"] = "Planned Parenthood"
                    
                    # Keep Link column as is for now - don't rename to URL yet
                    
                    # Keep only necessary columns in standard format
                    standard_columns = ["Question", "Answer", "Source", "Link"]
                    existing_columns = [col for col in standard_columns if col in df_pp.columns]
                    df_pp = df_pp[existing_columns]
                    
                    # Clean data
                    df_pp = df_pp.dropna(subset=["Question", "Answer"])
                    
                    # Add to collection
                    all_dfs.append(df_pp)
                    logger.info(f"Successfully loaded Planned Parenthood Data with {len(df_pp)} rows")
                else:
                    logger.warning(f"Planned Parenthood Data is missing required columns. Found: {df_pp.columns.tolist()}")
            else:
                logger.warning("Planned Parenthood Data CSV not found in either expected location")
            
            # Combine datasets
            if not all_dfs:
                logger.error("No datasets were successfully loaded.")
                return pd.DataFrame(columns=["Question", "Answer", "Source", "Link"])
            
            df_combined = pd.concat(all_dfs, ignore_index=True)
            
            # Final processing - preserve Link column
            for col in ["Question", "Answer", "Source", "Link"]:
                if col in df_combined.columns:
                    df_combined[col] = df_combined[col].astype(str).str.strip()
            
            # Filter out empty entries
            df_combined = df_combined[df_combined["Question"].str.len() > 0]
            df_combined = df_combined[df_combined["Answer"].str.len() > 0]
            
            # Replace 'nan' strings
            df_combined.replace('nan', '', inplace=True)
            
            # Final deduplication
            df_combined = df_combined.drop_duplicates(subset=["Question"], keep='first')
            
            # Now rename Link to URL at the very end
            if "Link" in df_combined.columns:
                df_combined = df_combined.rename(columns={"Link": "URL"})
                logger.info("Renamed 'Link' column to 'URL' in final combined dataframe")
            
            # Debug: Check final dataframe columns and URL values
            logger.info(f"Final combined dataset columns: {df_combined.columns.tolist()}")
            if "URL" in df_combined.columns:
                # Count non-empty URLs
                non_empty_urls = df_combined[df_combined["URL"].str.len() > 5].shape[0]
                logger.info(f"Entries with non-empty URLs: {non_empty_urls} out of {len(df_combined)}")
                # Sample URLs from final dataset
                sample_final_urls = df_combined["URL"].head(5).tolist()
                logger.info(f"Sample final URLs: {sample_final_urls}")
            
            logger.info(f"Loaded combined dataset with {len(df_combined)} unique entries")
            return df_combined
            
        except Exception as e:
            logger.error(f"Error in _load_datasets: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=["Question", "Answer", "Source", "URL"])


    def _build_index(self) -> Tuple[Optional[faiss.Index], Optional[np.ndarray]]:
        """
        Build FAISS index for efficient similarity search.

        Returns:
            Tuple[Optional[faiss.Index], Optional[np.ndarray]]: FAISS index and document embeddings, or None if failed.
        """
        if self.embedding_model is None or self.data.empty:
            logger.error("Cannot build index: embedding model or data not available.")
            return None, None

        try:
            # Use "Question" for indexing context
            texts = self.data["Question"].tolist()
            logger.info(f"Building index for {len(texts)} questions...")


            # Generate embeddings
            embeddings = self._generate_embeddings(texts)

            if embeddings is None or embeddings.size == 0:
                logger.error("Generated empty embeddings, cannot build index.")
                return None, None

            # Build FAISS index
            dimension = embeddings.shape[1]
            # Using IndexFlatL2 for Euclidean distance - lower score means more similar
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype(np.float32)) # Ensure float32

            logger.info(f"Built FAISS index with {index.ntotal} documents and dimension {dimension}")
            return index, embeddings

        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}", exc_info=True)
            return None, None


    def _generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a list of texts using the loaded sentence transformer.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            Optional[np.ndarray]: Embeddings array or None if failed.
        """
        if self.embedding_model is None or self.tokenizer is None:
            logger.error("Embedding model or tokenizer not available.")
            return None

        try:
            embeddings_list = []
            batch_size = 64 # Process in batches
            num_texts = len(texts)
            logger.info(f"Generating embeddings for {num_texts} texts in batches of {batch_size}...")

            # ***** FIX: Add timing for embedding generation *****
            embedding_start_time = time.time()
            # ***** END FIX *****

            for i in range(0, num_texts, batch_size):
                batch_texts = texts[i:min(i+batch_size, num_texts)]

                # Tokenize
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True,
                                       return_tensors="pt", max_length=512) # Standard max length

                # Generate embeddings
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)

                # Mean pooling
                attention_mask = inputs['attention_mask']
                mean_embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)

                # Normalize embeddings (best practice for distance metrics)
                normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
                embeddings_list.append(normalized_embeddings.cpu().numpy()) # Use normalized embeddings

                if (i + batch_size) % (batch_size * 10) == 0: # Log progress periodically
                     logger.debug(f"Processed {min(i+batch_size, num_texts)}/{num_texts} embeddings...")

            # ***** FIX: Log embedding generation time *****
            embedding_time = time.time() - embedding_start_time
            logger.info(f"Finished generating embeddings in {embedding_time:.4f} seconds.")
            # ***** END FIX *****
            
            return np.vstack(embeddings_list) if embeddings_list else None

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            return None


    def _mean_pooling(self, token_embeddings, attention_mask):
        """Perform mean pooling on token embeddings."""
        # Expand attention mask to match embeddings shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Sum embeddings where attention mask is 1
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # Compute the number of tokens (denominator), clamp to avoid division by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Return the mean
        return sum_embeddings / sum_mask


    async def _retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant documents for a query using FAISS index.

        Args:
            query (str): The query to search for.
            top_k (int): Number of documents to retrieve.

        Returns:
            Tuple[List[Dict[str, Any]], List[float]]: Retrieved documents and their distances.
        """
        if self.index is None or self.data.empty:
            logger.warning("Cannot retrieve context: FAISS index or data not available.")
            return [], []

        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])

            if query_embedding is None or query_embedding.size == 0:
                logger.warning("Generated empty query embedding, cannot search.")
                return [], []

            # Search the index
            k = min(top_k, self.index.ntotal) # Ensure k is not larger than index size
            if k <= 0:
                 logger.warning("Index is empty or k is zero, cannot search.")
                 return [], []

            # ***** FIX: Add timing for FAISS search *****
            search_start_time = time.time()
            # ***** END FIX *****

            # Perform the search using the query embedding
            # FAISS uses float32, ensure query embedding matches
            # Note: normalization is already handled in _generate_embeddings
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)

            # ***** FIX: Log FAISS search time *****
            search_time = time.time() - search_start_time
            logger.info(f"FAISS search completed in {search_time:.4f} seconds.")
            # ***** END FIX *****

            # Prepare results
            documents = []
            # Ensure distances and indices are valid numpy arrays before processing
            scores = distances[0].tolist() if isinstance(distances, np.ndarray) and distances.size > 0 else []
            valid_indices = indices[0] if isinstance(indices, np.ndarray) and indices.size > 0 else []


            if valid_indices.size > 0:
                 for i, idx in enumerate(valid_indices):
                     # Check index validity against the DataFrame size
                     if 0 <= idx < len(self.data):
                          row = self.data.iloc[idx].to_dict() # Convert row to dict safely
                          document = {
                              "question": row.get("Question", ""),
                              "answer": row.get("Answer", ""),
                              "source": row.get("Source", "Unknown Source"),
                              "url": row.get("URL", ""),
                              # Ensure score exists for the current index
                              "score": float(scores[i]) if i < len(scores) else float('inf')
                          }
                          documents.append(document)
                     else:
                          logger.warning(f"Retrieved invalid index {idx}. Index size: {self.index.ntotal}, Data size: {len(self.data)}")
            else:
                 logger.info("FAISS search returned no indices.")


            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}... with scores: {scores[:len(documents)]}")
            return documents, scores[:len(documents)] # Return scores matching documents

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return [], []


    def _combine_retrieved_docs(self, docs: List[Dict[str, Any]],
                               scores: List[float]) -> str:
        """
        Combine retrieved documents into a coherent response (RAG fallback).
        Prioritizes the most relevant document's answer.

        Args:
            docs (List[Dict[str, Any]]): Retrieved documents.
            scores (List[float]): Similarity scores for the documents.

        Returns:
            str: Combined response string.
        """
        if not docs:
            return "I don't have specific information about that topic. Could you please try asking another question about reproductive health?"

        # Use the most relevant document (lowest distance score) as the primary response
        primary_doc = docs[0]
        primary_answer = primary_doc.get('answer', 'Could not retrieve answer details.')

        # Simple approach: return the answer from the most relevant document
        # Future enhancement: Could potentially append a concise point from the second doc if distinct enough.
        logger.debug(f"RAG fallback: Using answer from best matching doc (score: {scores[0]:.4f})")
        return primary_answer


    def _extract_citations(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract formatted citations from retrieved documents.

        Args:
            docs (List[Dict[str, Any]]): List of retrieved document dictionaries.

        Returns:
            List[Dict[str, Any]]: List of unique citation objects.
        """
        unique_citations = {} # Use source+URL as key for uniqueness
        fallback_citations = {} # Store fallback citations without URLs
        
        logger.info(f"Extracting citations from {len(docs)} documents")
        
        # Debug: Log all document structures
        for i, doc in enumerate(docs):
            logger.debug(f"Document {i} keys: {doc.keys() if isinstance(doc, dict) else 'Not a dict'}")
            if isinstance(doc, dict) and 'url' in doc:
                logger.info(f"Document {i} has URL: {doc.get('url', '')[:50]}")
            elif isinstance(doc, dict) and 'URL' in doc:
                logger.info(f"Document {i} has URL: {doc.get('URL', '')[:50]}")
            
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict): 
                logger.debug(f"Skipping non-dict item at index {i}")
                continue # Skip non-dict items

            # Extract source, trying multiple possible field names
            source = doc.get('source', doc.get('Source', 'Unknown Source'))
            
            # Extract title, using question or title fields
            title = doc.get('question', doc.get('Question', doc.get('title', doc.get('Title', source))))
            
            # Extract URL, checking multiple possible field names (case-insensitive where possible)
            url = ''
            url_found = False
            for url_field in ['url', 'URL', 'link', 'Link']:
                if url_field in doc and doc[url_field] and isinstance(doc[url_field], str) and len(str(doc[url_field])) > 5:
                    url = doc[url_field].strip()
                    url_found = True
                    logger.info(f"Document {i} - Found URL in field '{url_field}': {url[:100]}")
                    break
            
            # Only use default URL as a fallback if no valid URL exists
            if not url_found and source == "Planned Parenthood":
                url = "https://www.plannedparenthood.org/"
                logger.info(f"Document {i} - Adding default URL for Planned Parenthood: {url}")
            
            # Log the extracted citation data
            logger.info(f"Document {i}: source='{source}', title='{title[:50]}...', url='{url[:50]}...'")

            # Create a citation object
            # Use a more descriptive title if possible
            display_title = f"{title}" if title != source else source
            if source != "Unknown Source" and title != source:
                display_title = f"{title} ({source})" # Combine title and source
            
            # Perform comprehensive URL validation
            has_valid_url = False
            if url and isinstance(url, str):
                # Basic check for http/https and non-placeholder domains
                parsed_url = urlparse(url)
                if parsed_url.scheme in ['http', 'https'] and parsed_url.netloc and '.' in parsed_url.netloc:
                    # Add more robust checks if needed to exclude specific placeholders
                    if parsed_url.netloc not in ['example.com', 'localhost', 'your.placeholder.domain']:
                        has_valid_url = True
                        logger.info(f"Document {i} - URL validated: {url[:50]}")
                    else:
                        logger.debug(f"Document {i} - Skipping placeholder domain URL: {url}")
                else:
                    logger.debug(f"Document {i} - Skipping invalid URL scheme/netloc: {url}")
            else:
                logger.debug(f"Document {i} - Skipping empty or non-string URL: {url}")
            
            # Only use fallback URL if no valid URL exists
            if source == "Planned Parenthood" and not has_valid_url:
                # Force a valid URL for Planned Parenthood
                url = "https://www.plannedparenthood.org/"
                logger.info(f"Document {i} - Forcing valid URL for Planned Parenthood citation: {url}")
            
            if has_valid_url:
                # Use source+URL as the key for deduplication to allow multiple URLs per source
                source_url_key = f"{source.lower()}:{url.lower().rstrip('/')}" 
                if source_url_key not in unique_citations:
                    citation_obj = {
                        "id": f"rag-{i+1}-{hash(source_url_key)}", # Create a unique ID
                        "source": source,
                        "url": url,
                        "title": display_title.strip(),
                        "accessed_date": datetime.now().strftime('%Y-%m-%d')
                    }
                    unique_citations[source_url_key] = citation_obj
                    logger.info(f"Created citation with source '{source}' and URL: {url[:50]}...")
            else:
                # Create a fallback citation without URL
                source_key = source.lower().strip()
                if source_key not in fallback_citations:
                    citation_obj = {
                        "id": f"rag-{i+1}-{hash(source_key)}", # Create a unique ID
                        "source": source,
                        "title": display_title.strip(),
                        "accessed_date": datetime.now().strftime('%Y-%m-%d')
                    }
                    fallback_citations[source_key] = citation_obj
                    logger.debug(f"Created fallback citation without URL for source: {source}")

        # Combine citations with valid URLs and fallback citations
        citations = list(unique_citations.values())
        logger.info(f"Created {len(citations)} citations with valid URLs")
        
        # Only add fallback citations if we have no citations with valid URLs
        if not citations and fallback_citations:
            citations = list(fallback_citations.values())
            logger.info(f"Using {len(citations)} fallback citations without URLs")
        
        # If we still have no citations, create a generic one
        if not citations and docs:
            generic_source = "Reproductive Health Information"
            citations = [{
                "id": f"rag-generic-{hash(generic_source)}",
                "source": generic_source,
                "title": "General Reproductive Health Information",
                "accessed_date": datetime.now().strftime('%Y-%m-%d')
            }]
            logger.info("Using generic citation as fallback")

        # Final verification of all citation URLs
        for i, citation in enumerate(citations):
            logger.info(f"Final citation {i}: source='{citation.get('source')}', url='{citation.get('url', '')[:50]}...'")

        return citations

# --- END OF FILE knowledge_handler.py ---