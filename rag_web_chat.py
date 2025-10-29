from langchain_ollama import OllamaLLM
from langchain_tavily import TavilySearch

llm = OllamaLLM(model="llama3.1:8b")
search = TavilySearch(k=3)

# Store conversation history
conversation_history = []
last_context = ""  # Store the last search context

while True:
    query = input("You: ").strip()
    if query.lower() in {"exit", "quit"}:
        break

    # Handle acknowledgments - just continue without doing anything
    acknowledgments = ["yes", "no", "ok", "okay", "cool", "thanks", "thank you", "got it", "sure", "alright", "k"]
    if query.lower() in acknowledgments:
        continue
    
    # Skip detection for obvious new questions (optimization)
    question_starters = ["what", "how", "why", "who", "when", "where", "which", "explain", "tell me", "describe"]
    is_obvious_question = any(query.lower().startswith(word) for word in question_starters)
    
    # Check if it's a modification command (no search needed)
    # Use LLM to detect if it's a modification request or new question
    if conversation_history and not is_obvious_question:
        last_topic = conversation_history[-1]['user']
        
        detection_prompt = f"""Previous question was about: "{last_topic}"

Current user input: "{query}"

Analyze if this is:
1. A MODIFICATION request - asking to adjust/rephrase the PREVIOUS response (e.g., "shorten that", "make it simpler", "rephrase")
2. A NEW_QUESTION - asking about a NEW topic or requesting NEW information (even if it includes style preferences like "explain X like I'm 10")

Key rule: If a NEW topic/subject is mentioned that's different from the previous question, it's a NEW_QUESTION.

Answer with ONLY "MODIFY" or "NEW_QUESTION". Nothing else.

Answer:"""
        
        detection_result = llm.invoke(detection_prompt).strip().upper()
        
        if "MODIFY" in detection_result:
            print("\n[Modifying previous response, no search needed]")
            
            # Build prompt for modification
            last_exchange = conversation_history[-1]
            
            modification_prompt = f"""Previous question: {last_exchange['user']}
Previous answer: {last_exchange['assistant']}

User's modification request: {query}

Please modify the previous answer according to the user's request. Use the same information but adjust the response style/length/complexity as requested."""

            answer = llm.invoke(modification_prompt)
            print(f"\nBot: {answer}\n")
            
            # Update the last entry in history instead of adding new one
            conversation_history[-1]['assistant'] = answer
            
            continue  # Skip the rest of the loop
    
    # Step 1: Reformulate query if needed based on conversation history
    if conversation_history and not is_obvious_question:
        reformulation_prompt = f"""Previous conversation:
{chr(10).join(f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in conversation_history[-2:])}

Current user question: {query}

If this question refers to something from the previous conversation (like "it", "that", "how does it work", etc.), rewrite it as a standalone search query that includes the full context. If it's already clear and standalone, return it exactly as-is.

IMPORTANT: Do not add context from previous topics unless the current question specifically refers to them.

Only output the rewritten query, nothing else. No explanations or extra text.

Rewritten search query:"""
        
        search_query = llm.invoke(reformulation_prompt).strip()
        
        # Clean up any extra text the LLM might add
        if '\n' in search_query:
            search_query = search_query.split('\n')[0].strip()
        
    else:
        search_query = query
    
    # Step 2: Enhance search query for better results
    query_lower = search_query.lower()
    
    # For "what is" questions - add intro keywords
    if any(phrase in query_lower for phrase in ["what is", "what are"]):
        enhanced_query = f"{search_query} introduction basics overview"
    
    # For "how/why" questions - add explanation/analysis keywords
    elif any(phrase in query_lower for phrase in ["how did", "why did", "how does", "why does", "how can", "why is"]):
        enhanced_query = f"{search_query} reasons explanation analysis"
    
    # For comparison questions
    elif any(phrase in query_lower for phrase in ["difference between", "compare", "vs", "versus"]):
        enhanced_query = f"{search_query} comparison detailed analysis"
    
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
    last_context = context  # Store for potential modifications
    
    # Step 4: Build prompt with conversation history (only last 2 exchanges to avoid style carryover)
    history_text = ""
    if conversation_history:
        history_text = "Previous conversation:\n"
        # Only include last 2 exchanges to prevent style from old questions carrying over
        for entry in conversation_history[-2:]:
            history_text += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
    
    # Check if user requested a specific style ONLY in current query
    style_instruction = ""
    if any(phrase in query.lower() for phrase in ["like i'm", "like im", "eli5", "explain like", "simply", "in simple"]):
        style_instruction = "\n\nIMPORTANT: Explain in a simple, easy-to-understand way as requested by the user in their current question."
    
    prompt = f"""{history_text}Current context from web search:
{context}

Current question: {query}{style_instruction}

Based on the conversation history (if any) and the current context, provide a helpful answer. Use a normal explanatory tone unless the user specifically requested a particular style in their current question."""

    # Step 5: Generate answer
    answer = llm.invoke(prompt)
    print(f"\nBot: {answer}\n")
    
    # Step 6: Save to conversation history
    conversation_history.append({
        "user": query,
        "assistant": answer
    })