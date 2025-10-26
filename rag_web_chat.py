from langchain_ollama import OllamaLLM
from langchain_tavily import TavilySearch

llm = OllamaLLM(model="llama3.1:8b")
search = TavilySearch(k=3)

# Store conversation history
conversation_history = []

while True:
    query = input("You: ").strip()
    if query.lower() in {"exit", "quit"}:
        break

    # Step 1: Reformulate query if needed based on conversation history
    if conversation_history:
        reformulation_prompt = f"""Previous conversation:
{chr(10).join(f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in conversation_history)}

Current user question: {query}

If this question refers to something from the previous conversation (like "it", "that", "how does it work", etc.), rewrite it as a standalone search query that includes the full context. If it's already clear and standalone, return it exactly as-is.

Only output the rewritten query, nothing else. No explanations or extra text.

Rewritten search query:"""
        
        search_query = llm.invoke(reformulation_prompt).strip()
        
        # Clean up any extra text the LLM might add
        if '\n' in search_query:
            search_query = search_query.split('\n')[0].strip()
        
    else:
        search_query = query
    
    # Step 2: Enhance search query for better results (add context keywords for intro questions)
    # Check if it's a "what is" type question
    query_lower = search_query.lower()
    if any(phrase in query_lower for phrase in ["what is", "what are", "explain", "introduction"]):
        enhanced_query = f"{search_query} introduction basics overview"
    else:
        enhanced_query = search_query
    
    print(f"\n[Searching for: {search_query}]")

    # Step 3: Search with the enhanced query
    results = search.invoke(enhanced_query)
    web_results = results.get("results", [])

    print("\nSources:")
    for r in web_results[:3]:
        print(f"- {r.get('title', 'No title')} | {r.get('url', 'No URL')}")

    context = "\n".join(r.get("content", "") for r in web_results[:3])
    
    # Step 4: Build prompt with conversation history
    history_text = ""
    if conversation_history:
        history_text = "Previous conversation:\n"
        for entry in conversation_history:
            history_text += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
    
    prompt = f"""{history_text}Current context from web search:
{context}

Current question: {query}

Based on the conversation history (if any) and the current context, provide a helpful answer:"""

    # Step 5: Generate answer
    answer = llm.invoke(prompt)
    print(f"\nBot: {answer}\n")
    
    # Step 6: Save to conversation history
    conversation_history.append({
        "user": query,
        "assistant": answer
    })