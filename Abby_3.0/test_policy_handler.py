import logging
import asyncio
from chatbot.policy_handler import PolicyHandler

logging.basicConfig(level=logging.DEBUG)

async def main():
    handler = PolicyHandler()
    
    # Test 1: State in original message should pass the sanity check
    print("Test 1: State in original message 'What is the abortion policy in Texas'")
    result1 = await handler.process_query(
        query="Tell me about Texas", 
        original_message="What is the abortion policy in Texas"
    )
    print(f"Test 1 Result: needs_state_info = {result1.get('needs_state_info')}, state_code = {result1.get('state_code')}")
    
    # Test 2: State not in original message should fail the sanity check
    print("\nTest 2: State not in original message 'What are the laws about abortion'")
    result2 = await handler.process_query(
        query="Tell me about TX", 
        original_message="What are the laws about abortion"
    )
    print(f"Test 2 Result: needs_state_info = {result2.get('needs_state_info')}, state_code = {result2.get('state_code', None)}")
    
    # Test 3: Original message with a different state
    print("\nTest 3: Different state in original message 'What about California?'")
    result3 = await handler.process_query(
        query="Tell me about TX", 
        original_message="What about California?"
    )
    print(f"Test 3 Result: needs_state_info = {result3.get('needs_state_info')}, state_code = {result3.get('state_code', None)}")

if __name__ == "__main__":
    asyncio.run(main()) 