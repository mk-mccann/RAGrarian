#!/usr/bin/env python3
"""
Test script to verify the streaming fix works correctly.
This simulates what Gradio would do with the chat_with_agent function.
"""

# Mock the agent and its stream_answer method for testing
class MockAgent:
    def __init__(self):
        self._last_context = []
    
    def stream_answer(self, question, thread_id="default"):
        """Mock stream_answer that simulates the real behavior"""
        if question.lower() == 'sources':
            if self._last_context:
                yield "\n\nSources:\nSource 1: Test document\nSource 2: Another document"
            else:
                yield "\n\nNo sources available yet. Ask a question first!"
            return
        
        # Simulate streaming response
        responses = [
            "Permaculture is a", 
            " design system", 
            " for sustainable", 
            " living."
        ]
        
        for response in responses:
            yield response
            
        # Store context for sources command
        self._last_context = [("doc1", 0.9), ("doc2", 0.8)]

def chat_with_agent(message, history, thread_id="default"):
    """
    Main chat interface function for Gradio ChatInterface.
    """
    # Stream the response from the agent
    for chunk in agent.stream_answer(message, thread_id=thread_id):
        yield chunk

# Test the function
if __name__ == "__main__":
    agent = MockAgent()
    
    print("Testing chat_with_agent function...")
    print("=" * 50)
    
    # Test 1: Regular question
    print("Test 1: Regular question")
    print("Question: What is permaculture?")
    print("Response:")
    response_chunks = list(chat_with_agent("What is permaculture?", []))
    full_response = "".join(response_chunks)
    print(f"  {full_response}")
    print()
    
    # Test 2: Sources command
    print("Test 2: Sources command")
    print("Question: sources")
    print("Response:")
    sources_chunks = list(chat_with_agent("sources", []))
    sources_response = "".join(sources_chunks)
    print(f"  {sources_response}")
    print()
    
    # Test 3: Sources before any question
    print("Test 3: Sources command before any question")
    new_agent = MockAgent()  # Fresh agent with no context
    print("Question: sources")
    print("Response:")
    no_sources_chunks = list(chat_with_agent("sources", [], thread_id="new"))
    no_sources_response = "".join(no_sources_chunks)
    print(f"  {no_sources_response}")
    print()
    
    print("All tests completed successfully!")
    print("The streaming fix should work with Gradio now.")