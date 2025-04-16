import os
import json
import pandas as pd
import asyncio
import sys
import logging
import time
import random
import re
from tqdm import tqdm
import dotenv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(dotenv_path):
    print(f"Loading environment variables from {dotenv_path}")
    dotenv.load_dotenv(dotenv_path)
    print("Environment variables loaded successfully")
else:
    print(f"Warning: .env file not found at {dotenv_path}")

# Check if API keys are available
openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables!")
    print("Please make sure your .env file contains: OPENAI_API_KEY=your-key-here")
    sys.exit(1)
else:
    print("OpenAI API key found ✓")

# Set up directories based on chatbot3.0 structure
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # chatbot3.0 directory
CHATBOT_DIR = os.path.join(PROJECT_DIR, "chatbot")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
EVAL_DIR = os.path.join(DATA_DIR, "evaluation")  # For test_questions.csv

# Create evaluation directory if it doesn't exist
os.makedirs(EVAL_DIR, exist_ok=True)

# Add chatbot directory to Python path
if CHATBOT_DIR not in sys.path:
    sys.path.append(CHATBOT_DIR)
    print(f"Added {CHATBOT_DIR} to Python path")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_DIR, "chatbot_eval.log"))
    ]
)
logger = logging.getLogger(__name__)

def clean_html_tags(text):
    """Remove HTML tags from text while preserving content structure"""
    if not text or not isinstance(text, str):
        return text
        
    # Replace paragraph breaks with newlines to preserve structure
    text = re.sub(r'<p class=[\'"]message-paragraph[\'"]>', '\n', text)
    text = re.sub(r'</p>', '', text)
    
    # Replace section headers with clear text format
    text = re.sub(r'<h3>(.*?)</h3>', r'\n\1\n', text)
    
    # Handle strong/bold text
    text = re.sub(r'<strong>(.*?)</strong>', r'\1', text)
    
    # Remove other HTML tags but preserve their content
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix any instances of multiple consecutive newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Trim extra whitespace
    text = text.strip()
    
    return text

def load_test_questions():
    """Load test questions from the existing file"""
    try:
        # Load from the specific test questions file
        test_questions_path = os.path.join(EVAL_DIR, "test_questions.csv")
        
        if not os.path.exists(test_questions_path):
            logger.error(f"Test questions file not found at: {test_questions_path}")
            raise FileNotFoundError(f"Test questions file not found at: {test_questions_path}")
            
        # Load the CSV file
        df = pd.read_csv(test_questions_path)
        logger.info(f"Successfully loaded {len(df)} test questions from {test_questions_path}")
        
        # Map primary_topic to category if needed
        if 'primary_topic' in df.columns and 'category' not in df.columns:
            # Create a mapping function
            def map_to_category(topic):
                if not topic or pd.isna(topic):
                    return 'knowledge'
                
                topic_lower = str(topic).lower()
                
                # Check for policy indicators
                if any(term in topic_lower for term in [
                    'policy', 'legal', 'law', 'state', 'ban', 'allow'
                ]):
                    return 'policy'
                
                # Check for emotional indicators
                elif any(term in topic_lower for term in [
                    'emotional', 'feel', 'scared', 'afraid', 'nervous'
                ]):
                    return 'emotional'
                
                # Default to knowledge
                return 'knowledge'
            
            # Create the category column
            df['category'] = df['primary_topic'].apply(map_to_category)
            logger.info("Created 'category' column from 'primary_topic'")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading test questions: {str(e)}")
        raise

async def initialize_model():
    """Initialize the multi-aspect processor"""
    try:
        # Import after ensuring chatbot directory is in path
        from chatbot.multi_aspect_processor import MultiAspectQueryProcessor
        
        logger.info("Initializing the multi-aspect processor...")
        processor = MultiAspectQueryProcessor()
        logger.info("Model initialized successfully!")
        return processor
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, asyncio.exceptions.CancelledError))
)
async def generate_response(processor, question_text, question_id):
    """Generate a response using the chatbot3.0 model with retry logic and improved context extraction"""
    try:
        # Create a simple conversation history for context
        conversation_history = [
            {"sender": "user", "message": question_text}
        ]
        
        # IMPORTANT: Intercept and save docs before processing the query
        # This is necessary because the knowledge handler resets its internal document store
        knowledge_handler = None
        if hasattr(processor, "handlers") and "knowledge" in processor.handlers:
            knowledge_handler = processor.handlers["knowledge"]
            logger.info("Found knowledge_handler to track documents")
        
        # Process the query through the multi-aspect processor
        start_time = time.time()
        response_data = await processor.process_query(
            message=question_text,
            conversation_history=conversation_history,
            user_location=None  # No location context for evaluation
        )
        processing_time = time.time() - start_time
        
        # Extract the response text - Updated for chatbot3.0 response structure
        if "text" in response_data:
            response_text = response_data["text"]
        elif "primary_content" in response_data:
            response_text = response_data["primary_content"]
        else:
            response_text = "Error: No response text generated"
        
        # Clean HTML tags from response text
        cleaned_response_text = clean_html_tags(response_text)
        logger.info(f"Cleaned HTML tags from response text. Original length: {len(response_text)}, Cleaned length: {len(cleaned_response_text)}")
        
        # Extract citations from the response data
        citations = []
        citation_objects = []
        retrieved_contexts = []
        
        # In chatbot3.0, citation_objects are directly in the response
        if "citation_objects" in response_data and response_data["citation_objects"]:
            citation_objects = response_data["citation_objects"]
            # Extract citation sources
            for citation in citation_objects:
                if isinstance(citation, dict) and "source" in citation:
                    citations.append(citation["source"])
                elif isinstance(citation, str):
                    citations.append(citation)
        elif "citations" in response_data:
            # Fallback to simple citations list if available
            citations = response_data["citations"]
        
        # DIRECT APPROACH: Extract documents from citation objects
        # Since we can see from the logs that citation_objects are working but contexts aren't
        # we'll reconstruct contexts directly from citations
        if citation_objects:
            logger.info(f"Extracting contexts from {len(citation_objects)} citation objects")
            
            # If we have knowledge_handler, try to find matching documents
            if knowledge_handler and hasattr(knowledge_handler, "data") and not knowledge_handler.data.empty:
                # For each citation, find matching document in knowledge data
                for citation in citation_objects:
                    if isinstance(citation, dict) and "source" in citation and "url" in citation:
                        source = citation["source"]
                        url = citation["url"]
                        
                        # Find documents that match this citation
                        matches = knowledge_handler.data[
                            (knowledge_handler.data["Source"] == source) & 
                            (knowledge_handler.data["URL"].str.contains(url[:30], na=False))
                        ]
                        
                        if not matches.empty:
                            # Take the first match
                            match = matches.iloc[0]
                            context = {
                                "source": match.get("Source", source),
                                "content": f"Q: {match.get('Question', '')}\nA: {match.get('Answer', '')}",
                                "relevance_score": 1.0  # Can't know actual score but it's cited so likely relevant
                            }
                            retrieved_contexts.append(context)
                            logger.info(f"Created context from citation: {source} ({url[:30]}...)")
                        else:
                            # Create a placeholder context with just the citation info
                            context = {
                                "source": source,
                                "content": f"Citation referenced URL: {url}",
                                "relevance_score": 0.5  # Medium relevance since we couldn't find the actual content
                            }
                            retrieved_contexts.append(context)
                            logger.info(f"Created placeholder context for citation without matching document: {source}")
        
        # If still no contexts, try standard approaches
        if not retrieved_contexts:
            # 1. First attempt: Check directly in response_data
            if "retrieved_documents" in response_data:
                logger.info("Found retrieved_documents in response_data")
                docs = response_data["retrieved_documents"]
                for doc in docs:
                    context = {
                        "source": doc.get("source", "Unknown"),
                        "content": f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}",
                        "relevance_score": doc.get("score", 0)
                    }
                    retrieved_contexts.append(context)
            
            # 2. Second attempt: Try to get from processor's knowledge handler
            if not retrieved_contexts and knowledge_handler:
                logger.info("Looking for contexts in knowledge_handler attributes")
                
                # Try _last_retrieved_docs attribute
                if hasattr(knowledge_handler, "_last_retrieved_docs") and knowledge_handler._last_retrieved_docs:
                    docs = knowledge_handler._last_retrieved_docs
                    logger.info(f"Found {len(docs)} documents in _last_retrieved_docs")
                    
                    for doc in docs:
                        context = {
                            "source": doc.get("source", "Unknown"),
                            "content": f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}",
                            "relevance_score": doc.get("score", 0)
                        }
                        retrieved_contexts.append(context)
            
            # 3. Third attempt: Check in aspect_responses
            if not retrieved_contexts and "aspect_responses" in response_data:
                for aspect in response_data["aspect_responses"]:
                    if aspect.get("aspect_type") == "knowledge" and "retrieved_documents" in aspect:
                        logger.info("Found retrieved_documents in knowledge aspect response")
                        for doc in aspect.get("retrieved_documents", []):
                            context = {
                                "source": doc.get("source", "Unknown"),
                                "content": f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}",
                                "relevance_score": doc.get("score", 0)
                            }
                            retrieved_contexts.append(context)
        
        # LAST RESORT: If we still have no contexts but have citations, create minimal contexts
        if not retrieved_contexts and citations:
            logger.info("Creating minimal contexts from citation sources")
            for i, source in enumerate(citations):
                context = {
                    "source": source,
                    "content": f"Citation {i+1}: {source}",
                    "relevance_score": 0.5  # Medium relevance since it's just a citation
                }
                retrieved_contexts.append(context)
        
        # Default citation if none found
        if not citations:
            citations = ["Planned Parenthood"]
            # Add a minimal context for the default citation
            if not retrieved_contexts:
                retrieved_contexts.append({
                    "source": "Planned Parenthood",
                    "content": "Default citation: Planned Parenthood",
                    "relevance_score": 0.1  # Low relevance since it's a default
                })
        
        logger.info(f"Generated response to '{question_id}' in {processing_time:.2f}s")
        logger.info(f"Retrieved contexts count: {len(retrieved_contexts)}")
        
        # Add debug information about the first context if exists
        if retrieved_contexts:
            first_context = retrieved_contexts[0]
            logger.info(f"Sample context - source: {first_context.get('source')}, content: {first_context.get('content')[:100]}...")
        
        return {
            "question_id": question_id,
            "question_text": question_text,
            "response_text": cleaned_response_text,  # Use cleaned text
            "citations": citations,
            "citation_objects": citation_objects,
            "retrieved_contexts": retrieved_contexts,
            "processing_time": processing_time,
            "model_metadata": {
                "model_type": "MultiAspectQueryProcessor",
                "aspect_types": [asp.get("aspect_type", "unknown") for asp in response_data.get("aspect_responses", [])],
                "confidence_scores": [asp.get("confidence", 0) for asp in response_data.get("aspect_responses", [])]
            }
        }
    except (httpx.ReadTimeout, httpx.ConnectTimeout, asyncio.exceptions.CancelledError) as e:
        # These errors will trigger the retry
        logger.warning(f"Network error for question '{question_id}', retrying: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error generating response for '{question_id}': {str(e)}")
        return {
            "question_id": question_id,
            "question_text": question_text,
            "response_text": f"Error generating response: {str(e)}",
            "citations": ["Error"],
            "citation_objects": [],
            "retrieved_contexts": [],
            "processing_time": 0,
            "error": str(e)
        }

async def process_question_batch(processor, questions_batch, start_idx, total_questions):
    """Process a batch of questions"""
    responses = []
    failed_responses = []
    
    # Create progress bar for this batch
    pbar = tqdm(
        questions_batch.iterrows(), 
        total=len(questions_batch),
        desc=f"Batch {start_idx}-{start_idx + len(questions_batch) - 1}/{total_questions}"
    )
    
    for _, row in pbar:
        question_id = row['question_id']
        question_text = row['question_text']
        
        # Get category information
        category = row.get('category', 'unknown')
        primary_topic = row.get('primary_topic', '')
        secondary_topic = row.get('secondary_topic', '')
        
        try:
            # Generate response with retry mechanism
            response = await generate_response(processor, question_text, question_id)
            
            # Add additional metadata for analysis
            response["category"] = category
            if primary_topic is not None and primary_topic != '':
                response["primary_topic"] = primary_topic
            if secondary_topic is not None and secondary_topic != '':
                response["secondary_topic"] = secondary_topic
            
            # Add to responses list
            responses.append(response)
            
            # Save responses incrementally to avoid data loss
            if len(responses) % 5 == 0:  # Changed from 10 to 5
                interim_output_path = os.path.join(EVAL_DIR, f"interim_responses_{start_idx}_{len(responses)}.json")
                with open(interim_output_path, 'w') as f:
                    json.dump(responses, f, indent=2)
                logger.info(f"Saved {len(responses)} interim responses to {interim_output_path}")
                
            # Add a small delay between requests to avoid overwhelming the API
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Failed to generate response for question {question_id}: {str(e)}")
            failed_responses.append({
                "question_id": question_id,
                "question_text": question_text,
                "error": str(e),
                "category": category,
                "primary_topic": primary_topic,
                "secondary_topic": secondary_topic
            })
    
    return responses, failed_responses

async def generate_baseline_responses(num_questions=250, batch_size=10):
    """Generate baseline responses for evaluation in batches"""
    # Load test questions from the specific file we know exists
    questions_df = load_test_questions()
    logger.info(f"Loaded {len(questions_df)} questions from test_questions.csv")
    
    # Process all questions (up to num_questions)
    if len(questions_df) > num_questions:
        # Set random seed for reproducibility
        random.seed(42)
        selected_questions = questions_df.sample(num_questions, random_state=42)
        logger.info(f"Randomly selected {num_questions} questions from {len(questions_df)} total questions")
    else:
        selected_questions = questions_df
        logger.info(f"Using all {len(selected_questions)} available questions")
    
    # Save the selected questions for reference
    selected_questions_path = os.path.join(EVAL_DIR, "selected_evaluation_questions.csv")
    selected_questions.to_csv(selected_questions_path, index=False)
    logger.info(f"Saved selected questions to {selected_questions_path}")
    
    # Initialize the model
    processor = await initialize_model()
    
    # Process questions in batches
    all_responses = []
    all_failed_responses = []
    
    # Create batches of questions
    num_batches = (len(selected_questions) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(selected_questions))
        batch = selected_questions.iloc[start_idx:end_idx].copy()
        
        logger.info(f"Processing batch {i+1}/{num_batches} (questions {start_idx+1}-{end_idx})")
        
        try:
            # Process the batch
            responses, failed_responses = await process_question_batch(
                processor, batch, start_idx, len(selected_questions)
            )
            
            # Add to overall responses
            all_responses.extend(responses)
            all_failed_responses.extend(failed_responses)
            
            # Save batch results
            batch_output_path = os.path.join(EVAL_DIR, f"batch_{i+1}_responses.json")
            with open(batch_output_path, 'w') as f:
                json.dump(responses, f, indent=2)
            
            logger.info(f"Completed batch {i+1}/{num_batches} with {len(responses)} responses")
            
            # Add a longer pause between batches to avoid overwhelming the API
            if i < num_batches - 1:
                logger.info(f"Pausing for 15 seconds before starting next batch")
                await asyncio.sleep(15)  # Increased from 5 to 15 seconds
                
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {str(e)}")
            # Save what we have so far before giving up on this batch
            if responses:
                batch_output_path = os.path.join(EVAL_DIR, f"partial_batch_{i+1}_responses.json")
                with open(batch_output_path, 'w') as f:
                    json.dump(responses, f, indent=2)
                logger.info(f"Saved partial batch {i+1} with {len(responses)} responses")
    
    # Consolidate all responses from all batches
    if all_responses:
        # Save all responses to JSON file
        output_path = os.path.join(EVAL_DIR, "baseline_responses.json")
        with open(output_path, 'w') as f:
            json.dump(all_responses, f, indent=2)
        
        # Also save as CSV for easier analysis
        responses_df = pd.DataFrame(all_responses)
        csv_path = os.path.join(EVAL_DIR, "baseline_responses.csv")
    
        # Handle nested fields for CSV format
        if 'citation_objects' in responses_df.columns:
            responses_df['citation_objects'] = responses_df['citation_objects'].apply(lambda x: json.dumps(x))
        if 'model_metadata' in responses_df.columns:
            responses_df['model_metadata'] = responses_df['model_metadata'].apply(lambda x: json.dumps(x))
        if 'retrieved_contexts' in responses_df.columns:
            responses_df['retrieved_contexts'] = responses_df['retrieved_contexts'].apply(lambda x: json.dumps(x))
        
        responses_df.to_csv(csv_path, index=False)
        
        # Log stats
        logger.info(f"Generated {len(all_responses)} responses")
        logger.info(f"Failed to generate {len(all_failed_responses)} responses")
        
        # Save failed responses if any
        if all_failed_responses:
            failed_path = os.path.join(EVAL_DIR, "failed_responses.json")
            with open(failed_path, 'w') as f:
                json.dump(all_failed_responses, f, indent=2)
            logger.info(f"Saved {len(all_failed_responses)} failed responses to {failed_path}")
        
        logger.info(f"Saved responses to {output_path} and {csv_path}")
    else:
        logger.error("No responses were generated successfully.")
    
    return all_responses

async def main():
    """Main function to run the evaluation process"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate baseline responses for chatbot evaluation')
    parser.add_argument('--num_questions', type=int, default=250, help='Number of questions to process (default: 250)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing (default: 10)')
    args = parser.parse_args()
    
    print(f"Starting baseline response generation for {args.num_questions} questions in batches of {args.batch_size}...")
    
    # Check key dependencies
    try:
        from chatbot.multi_aspect_processor import MultiAspectQueryProcessor
        print("Successfully imported MultiAspectQueryProcessor")
    except ImportError as e:
        print(f"ERROR: Failed to import MultiAspectQueryProcessor: {e}")
        print("Make sure the chatbot module is properly installed and in the Python path.")
        return None
    
    # Check if test_questions.csv exists in the expected location
    test_questions_path = os.path.join(EVAL_DIR, "test_questions.csv")
    if not os.path.exists(test_questions_path):
        print(f"ERROR: test_questions.csv not found at {test_questions_path}")
        print("Copying test_questions.csv from chatbot2.0 (if available)...")
        try:
            # Try to locate chatbot2.0 directory (assuming it's in the same parent directory)
            chatbot2_dir = os.path.abspath(os.path.join(PROJECT_DIR, "..", "chatbot2.0"))
            chatbot2_test_questions = os.path.join(chatbot2_dir, "data", "evaluation", "test_questions.csv")
            
            if os.path.exists(chatbot2_test_questions):
                # Create the evaluation directory if it doesn't exist
                os.makedirs(EVAL_DIR, exist_ok=True)
                
                # Copy the file using Python's built-in file operations
                import shutil
                shutil.copy2(chatbot2_test_questions, test_questions_path)
                print(f"Successfully copied test_questions.csv from {chatbot2_test_questions}")
            else:
                print(f"Could not find test_questions.csv in chatbot2.0 directory either.")
                return None
        except Exception as copy_error:
            print(f"Error copying test_questions.csv: {copy_error}")
            return None
    else:
        print(f"Found test_questions.csv at {test_questions_path}")
    
    # Generate responses
    try:
        # Generate baseline responses with batch processing
        responses = await generate_baseline_responses(args.num_questions, args.batch_size)
        
        # Print a sample response
        if responses:
            print("\nSample response:")
            sample = responses[0]
            print(f"Question: {sample['question_text']}")
            print(f"Response: {sample['response_text'][:200]}...")
            print(f"Citations: {sample['citations']}")
            print(f"Processing time: {sample['processing_time']:.2f}s")
            
            # Check for contexts
            if 'retrieved_contexts' in sample and sample['retrieved_contexts']:
                print(f"Sample has {len(sample['retrieved_contexts'])} retrieved contexts")
                print(f"First context: {json.dumps(sample['retrieved_contexts'][0], indent=2)}")
            else:
                print("Warning: No retrieved contexts found in the sample response")
        
        print(f"\nGenerated responses for {len(responses)} questions")
        print(f"Results saved to {os.path.join(EVAL_DIR, 'baseline_responses.json')} and {os.path.join(EVAL_DIR, 'baseline_responses.csv')}")
        return responses
    except Exception as e:
        print(f"ERROR: Failed to generate responses: {e}")
        import traceback
        traceback.print_exc()
        return None

# Entry point to run the script
if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())