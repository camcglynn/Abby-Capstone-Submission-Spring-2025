import logging
import asyncio
from chatbot.policy_handler import PolicyHandler

logging.basicConfig(level=logging.DEBUG)

async def main():
    handler = PolicyHandler()
    
    # Test Policy Formatter with New Mexico data
    nm_data = {
        "abortion_legal": True,
        "gestational_limit": None,  # Specific limit not available
        "waiting_period": None,  # Information not available
        "minors_access": "No parental involvement required",
        "state_code": "NM"
    }
    
    # Test 1: Format directly with the formatter
    print("Test 1: Direct formatter output for New Mexico")
    formatter_result = handler._format_policy_data("NM", nm_data)
    print(f"Formatter Result:\n{formatter_result}\n")
    
    # Test 2: Generate with OpenAI
    print("Test 2: OpenAI generation for New Mexico")
    openai_result = await handler._generate_with_openai("Tell me about New Mexico", "NM", nm_data)
    print(f"OpenAI Result:\n{openai_result}\n")
    
    # Test 3: Full process_query
    print("Test 3: Full process_query for New Mexico")
    query_result = await handler.process_query(
        query="Is abortion legal in New Mexico?", 
        original_message="Is abortion legal in New Mexico?"
    )
    print(f"Process Query Result Text:\n{query_result.get('text', 'No text found')}\n")
    
    # Add additional logging to help debug any issues
    print("Debug Information:")
    print(f"State Code in Result: {query_result.get('state_code')}")
    print(f"Needs State Info: {query_result.get('needs_state_info')}")
    print(f"Question Answered: {query_result.get('question_answered')}")
    
    # To verify the content is preserved end-to-end, test with a direct message too
    print("\nTest 4: Direct message through process_query")
    direct_result = await handler.process_query(
        query="New Mexico", 
        original_message="New Mexico"
    )
    print(f"Direct Message Result:\n{direct_result.get('text', 'No text found')}")

if __name__ == "__main__":
    asyncio.run(main()) 