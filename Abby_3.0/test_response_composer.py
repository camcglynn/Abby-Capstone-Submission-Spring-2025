import logging
import asyncio
from chatbot.response_composer import ResponseComposer

logging.basicConfig(level=logging.DEBUG)

async def main():
    composer = ResponseComposer()
    
    # Test 1: Text with stray > characters at beginning of lines
    test1_text = "> This is a line with a leading >\n>Another line with >\n> A third line"
    print("Test 1: Text with stray > at beginning of lines")
    result1 = await composer._apply_final_formatting_and_cleanup(test1_text)
    print(f"Original: {test1_text}")
    print(f"Cleaned: {result1}")
    
    # Test 2: Text with stray > characters mid-line
    test2_text = "This is a line with a > mid-line character\nAnother line with mid > line character"
    print("\nTest 2: Text with stray > in middle of lines")
    result2 = await composer._apply_final_formatting_and_cleanup(test2_text)
    print(f"Original: {test2_text}")
    print(f"Cleaned: {result2}")
    
    # Test 3: Text with Markdown elements that should be preserved
    test3_text = "> **Heading**\n> • List item 1\n> • List item 2"
    print("\nTest 3: Text with markdown elements to preserve")
    result3 = await composer._apply_final_formatting_and_cleanup(test3_text)
    print(f"Original: {test3_text}")
    print(f"Cleaned: {result3}")

if __name__ == "__main__":
    asyncio.run(main()) 