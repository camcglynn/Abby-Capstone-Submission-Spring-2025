import logging
import asyncio
from chatbot.unified_classifier import UnifiedClassifier

logging.basicConfig(level=logging.DEBUG)

async def main():
    classifier = UnifiedClassifier()
    
    # Test 1: Query with explicit state name should detect location
    print("Test 1: Query with explicit state 'What is the abortion policy in Texas?'")
    result1 = await classifier.classify("What is the abortion policy in Texas?")
    print(f"Test 1 Result: contains_location = {result1.get('contains_location')}, detected_locations = {result1.get('detected_locations', [])}")
    
    # Test 2: Query with 2-letter abbreviation should detect location
    print("\nTest 2: Query with 2-letter abbreviation 'What's the law in TX?'")
    result2 = await classifier.classify("What's the law in TX?")
    print(f"Test 2 Result: contains_location = {result2.get('contains_location')}, detected_locations = {result2.get('detected_locations', [])}")
    
    # Test 3: Query with ZIP code should detect location
    print("\nTest 3: Query with ZIP code 'Are abortions available near 90210?'")
    result3 = await classifier.classify("Are abortions available near 90210?")
    print(f"Test 3 Result: contains_location = {result3.get('contains_location')}, detected_locations = {result3.get('detected_locations', [])}")
    
    # Test 4: Query without location shouldn't detect any
    print("\nTest 4: Query without location 'What are the abortion laws?'")
    result4 = await classifier.classify("What are the abortion laws?")
    print(f"Test 4 Result: contains_location = {result4.get('contains_location')}, detected_locations = {result4.get('detected_locations', [])}")
    
    # Test 5: Query with "my state" should detect location flag but not specific location
    print("\nTest 5: Query with 'my state' 'What are the abortion laws in my state?'")
    result5 = await classifier.classify("What are the abortion laws in my state?")
    print(f"Test 5 Result: contains_location = {result5.get('contains_location')}, detected_locations = {result5.get('detected_locations', [])}")

if __name__ == "__main__":
    asyncio.run(main()) 