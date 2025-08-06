from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType

# 1. Load local model (Ollama)
llm = Ollama(model="llama3.1:8b")

# 2. Load web search tool (Tavily)
search = TavilySearchResults(k=3)

# 3. Create agent with search + math + llm
tools = [search]
'''
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=6,
    max_execution_time=60
)
'''

chat_history = []  # Stores previous Q&A turns

# 4. Chat loop
print("Ask anything (type 'exit' to quit):")
while True:
    query = input("You: ")
    if query.strip().lower() == "exit":
        break

    # 1. Run search manually
    search_results = search._run(query)  # returns a list of dicts

    # 2. Extract top 3 snippet contents
    # Flatten any nested lists and filter dicts with "content"
    flattened = []
    for item in search_results:
        if isinstance(item, list):
            flattened.extend([x for x in item if isinstance(x, dict)])
        elif isinstance(item, dict):
            flattened.append(item)

    snippets = [r["content"] for r in flattened if "content" in r][:3]

    # Get and display sources
    sources = [
    f"- {r.get('title', 'No title')} | {r.get('url', 'No URL')}"
    for r in flattened if "content" in r
    ][:3]

    if sources:
        print("Sources:")
        for src in sources:
            print(src)
    else:
        print("No sources to show.")
    
    # Debug to check if snippets is full or empty
    print("search_results type:", type(flattened))
    print("Types of items inside search_results:", [type(r) for r in flattened])


    # Build conversation history (last 2 turns)
    history_text = "\n".join([
        f"Q: {turn['question']}\nA: {turn['answer']}"
        for turn in chat_history[-2:]
    ])

    # Full prompt with memory + web context
    prompt = f"""You are a helpful assistant. Use the previous conversation and the web context to answer the user's new question.

    Conversation history:
    {history_text if history_text else 'None yet'}

    Current question: {query}

    Context:
    {snippets[0] if len(snippets) > 0 else ''}
    {snippets[1] if len(snippets) > 1 else ''}
    {snippets[2] if len(snippets) > 2 else ''}

    Answer:"""


    # 4. Ask the LLM
    response = llm.invoke(prompt)
    print(f"\nBot: {response}\n")

    chat_history.append({
    "question": query,
    "answer": str(response).strip()
    })

    