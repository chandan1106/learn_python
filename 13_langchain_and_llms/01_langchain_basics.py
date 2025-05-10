"""
LangChain Basics: Introduction to LangChain Framework
"""

# This file provides an introduction to LangChain for building LLM applications
# In a real environment, you would need to install LangChain: pip install langchain
print("Note: This code assumes LangChain is installed. If you get an ImportError, install it with: pip install langchain")

# ===== INTRODUCTION TO LANGCHAIN =====
print("\n===== INTRODUCTION TO LANGCHAIN =====")
"""
LangChain is a framework for developing applications powered by language models. It enables:
1. Context-aware reasoning
2. Connecting LLMs to external data sources
3. Allowing LLMs to interact with their environment
4. Building complex chains and agents

Key Components:
1. Models: Interfaces to language models (LLMs, chat models, text embedding models)
2. Prompts: Templates and management for model inputs
3. Memory: State persistence between chain or agent calls
4. Indexes: Tools for structuring documents for efficient LLM interaction
5. Chains: Sequences of operations for specific tasks
6. Agents: LLMs that can use tools and make decisions
"""

try:
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    print("LangChain successfully imported!")
except ImportError:
    print("LangChain is not installed. Please install it with: pip install langchain")
    print("Continuing with code examples that you can run after installing...")

# ===== WORKING WITH LANGUAGE MODELS =====
print("\n===== WORKING WITH LANGUAGE MODELS =====")
"""
LangChain provides a unified interface to various LLMs:
- OpenAI (GPT models)
- Anthropic (Claude)
- Hugging Face models
- Local models (e.g., LLaMA, Falcon)

Example (requires API key):
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Text completion model
llm = OpenAI(temperature=0.7)
result = llm("What is the capital of France?")

# Chat model
chat_model = ChatOpenAI()
from langchain.schema import HumanMessage
result = chat_model([HumanMessage(content="What is the capital of France?")])
```
"""

# Example code (commented out as it requires API keys)
"""
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"  # Never hardcode API keys in production

from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)
result = llm("Write a short poem about Python programming")
print(result)
"""

print("Example of using OpenAI with LangChain (requires API key):")
print('''
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"  # Set via environment variable in production

from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)
result = llm("Write a short poem about Python programming")
print(result)
''')

# ===== PROMPT TEMPLATES =====
print("\n===== PROMPT TEMPLATES =====")
"""
Prompt Templates help create dynamic prompts with variables.
"""

print("Example of using PromptTemplate:")
print('''
from langchain.prompts import PromptTemplate

# Create a prompt template
template = "Write a {length} summary of the following text: {text}"
prompt = PromptTemplate(
    input_variables=["length", "text"],
    template=template,
)

# Format the prompt with specific values
formatted_prompt = prompt.format(
    length="short",
    text="LangChain is a framework for developing applications powered by language models."
)
print(formatted_prompt)

# Use it with an LLM
result = llm(formatted_prompt)
print(result)
''')

# ===== CHAINS =====
print("\n===== CHAINS =====")
"""
Chains combine multiple components (LLMs, prompts, etc.) for specific tasks.
"""

print("Example of using LLMChain:")
print('''
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Create a prompt template
template = "Write a {length} summary of the following text: {text}"
prompt = PromptTemplate(input_variables=["length", "text"], template=template)

# Create an LLM
llm = OpenAI(temperature=0.7)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run(
    length="short",
    text="LangChain is a framework for developing applications powered by language models."
)
print(result)
''')

# ===== MEMORY =====
print("\n===== MEMORY =====")
"""
Memory components store and manage the state between chain or agent calls.
"""

print("Example of using ConversationBufferMemory:")
print('''
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=memory,
    verbose=True
)

# First interaction
response1 = conversation.predict(input="Hi, my name is Alice")
print(response1)

# Second interaction (the model remembers the previous interaction)
response2 = conversation.predict(input="What's my name?")
print(response2)
''')

# ===== DOCUMENT LOADERS AND TEXT SPLITTERS =====
print("\n===== DOCUMENT LOADERS AND TEXT SPLITTERS =====")
"""
Document loaders help import documents from various sources.
Text splitters divide documents into manageable chunks.
"""

print("Example of loading and splitting documents:")
print('''
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load document
loader = TextLoader("path/to/document.txt")
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks")
''')

# ===== EMBEDDINGS AND VECTOR STORES =====
print("\n===== EMBEDDINGS AND VECTOR STORES =====")
"""
Embeddings convert text to vector representations.
Vector stores index these for semantic search.
"""

print("Example of creating embeddings and using a vector store:")
print('''
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load and split the document
loader = TextLoader("path/to/document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings and store in vector database
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# Query the database
query = "What is LangChain?"
docs = db.similarity_search(query)
print(docs[0].page_content)
''')

# ===== RETRIEVAL AUGMENTED GENERATION (RAG) =====
print("\n===== RETRIEVAL AUGMENTED GENERATION (RAG) =====")
"""
RAG combines retrieval of relevant documents with LLM generation.
"""

print("Example of a simple RAG system:")
print('''
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Load and process the document
loader = TextLoader("path/to/document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# Create a retrieval chain
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

# Ask a question
query = "What is the main topic of this document?"
result = qa.run(query)
print(result)
''')

# ===== BUILDING AGENTS =====
print("\n===== BUILDING AGENTS =====")
"""
Agents use LLMs to determine which actions to take and in what order.
They can use tools and make decisions based on the task.
"""

print("Example of creating an agent with tools:")
print('''
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(temperature=0)

# Load tools (e.g., search, calculator)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Initialize agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What was the high temperature in SF yesterday? What is that number raised to the 0.023 power?")
print(result)
''')

# ===== CUSTOM TOOLS FOR AGENTS =====
print("\n===== CUSTOM TOOLS FOR AGENTS =====")
"""
You can create custom tools for agents to use.
"""

print("Example of creating a custom tool:")
print('''
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import OpenAI

def get_weather(location):
    """Get the weather in a location"""
    # In a real scenario, you would call a weather API here
    return f"The weather in {location} is sunny and 75 degrees"

# Create a tool
weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Useful for getting the weather in a specific location"
)

# Initialize the agent with the custom tool
llm = OpenAI(temperature=0)
agent = initialize_agent(
    [weather_tool], 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What's the weather like in San Francisco?")
print(result)
''')

# ===== BUILDING A CHATBOT WITH LANGCHAIN =====
print("\n===== BUILDING A CHATBOT WITH LANGCHAIN =====")
"""
LangChain makes it easy to build chatbots with memory.
"""

print("Example of a simple chatbot:")
print('''
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the chat model
chat = ChatOpenAI(temperature=0.7)

# Create memory
memory = ConversationBufferMemory()

# Create a conversation chain
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

# Chat loop
print("Chatbot: Hello! How can I help you today? (type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = conversation.predict(input=user_input)
    print(f"Chatbot: {response}")
''')

# ===== PRACTICAL EXAMPLE: DOCUMENT Q&A SYSTEM =====
print("\n===== PRACTICAL EXAMPLE: DOCUMENT Q&A SYSTEM =====")
"""
A complete example of a document Q&A system using LangChain.
"""

print("Example of a document Q&A system:")
print('''
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory

# Set API key
os.environ["OPENAI_API_KEY"] = "your-api-key"  # Set via environment variable in production

# Load documents from a directory
loader = DirectoryLoader("./documents/", glob="**/*.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the conversational chain
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Chat loop
print("Document Q&A System: Ask questions about your documents (type 'exit' to quit)")
while True:
    query = input("You: ")
    if query.lower() == 'exit':
        break
    result = qa({"question": query})
    print(f"Answer: {result['answer']}")
''')

# ===== ADVANCED LANGCHAIN PATTERNS =====
print("\n===== ADVANCED LANGCHAIN PATTERNS =====")
"""
Advanced patterns and techniques in LangChain.
"""

print("1. Chain of Thought Prompting:")
print('''
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

template = """
Question: {question}

Let's think through this step by step:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAI(temperature=0)

question = "If I have 5 apples and give 2 to my friend, then buy 3 more, how many apples do I have?"
chain_of_thought_prompt = prompt.format(question=question)
result = llm(chain_of_thought_prompt)
print(result)
''')

print("\n2. Self-Critique and Refinement:")
print('''
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain

# First chain to generate an answer
answer_template = "Question: {question}\n\nAnswer:"
answer_prompt = PromptTemplate(template=answer_template, input_variables=["question"])
answer_chain = LLMChain(llm=OpenAI(temperature=0), prompt=answer_prompt, output_key="answer")

# Second chain to critique the answer
critique_template = "Answer: {answer}\n\nCritique this answer and identify any mistakes or areas for improvement:"
critique_prompt = PromptTemplate(template=critique_template, input_variables=["answer"])
critique_chain = LLMChain(llm=OpenAI(temperature=0), prompt=critique_prompt, output_key="critique")

# Third chain to refine the answer based on critique
refine_template = "Original answer: {answer}\n\nCritique: {critique}\n\nImproved answer:"
refine_prompt = PromptTemplate(template=refine_template, input_variables=["answer", "critique"])
refine_chain = LLMChain(llm=OpenAI(temperature=0), prompt=refine_prompt, output_key="refined_answer")

# Combine the chains
self_critique_chain = SequentialChain(
    chains=[answer_chain, critique_chain, refine_chain],
    input_variables=["question"],
    output_variables=["answer", "critique", "refined_answer"],
    verbose=True
)

result = self_critique_chain({"question": "Explain the theory of relativity"})
print(f"Final answer: {result['refined_answer']}")
''')

print("\n===== CONCLUSION =====")
print("""
LangChain provides a powerful framework for building LLM applications:
1. It simplifies working with LLMs through a unified interface
2. It provides tools for context management, memory, and document processing
3. It enables the creation of agents that can use tools and make decisions
4. It facilitates the development of complex applications like chatbots and Q&A systems

To get started with LangChain:
1. Install it: pip install langchain
2. Set up API keys for your preferred LLM provider
3. Start with simple chains and gradually build more complex applications
4. Explore the documentation and examples at https://python.langchain.com/
""")

print("\n===== END OF LANGCHAIN BASICS =====")