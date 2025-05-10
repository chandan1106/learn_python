"""
Building LLM Applications: Advanced Techniques and Best Practices
"""

# This file covers advanced techniques for building LLM applications
# In a real environment, you would need to install: pip install langchain openai tiktoken

# ===== INTRODUCTION =====
print("\n===== INTRODUCTION TO LLM APPLICATIONS =====")
"""
Large Language Models (LLMs) can be used to build a wide variety of applications:
1. Chatbots and virtual assistants
2. Content generation and summarization
3. Information extraction and analysis
4. Code generation and explanation
5. Question answering systems
6. And much more

This guide covers advanced techniques and best practices for building robust LLM applications.
"""

# ===== PROMPT ENGINEERING =====
print("\n===== PROMPT ENGINEERING =====")
"""
Prompt engineering is the process of designing effective prompts to get the best results from LLMs.

Key Techniques:

1. Zero-shot prompting:
   - Ask the model to perform a task without examples
   - Example: "Translate the following English text to French: {text}"

2. Few-shot prompting:
   - Provide a few examples before asking the model to perform a task
   - Example: 
     "English: Hello
      French: Bonjour
      English: How are you?
      French: Comment allez-vous?
      English: {text}
      French:"

3. Chain of Thought (CoT):
   - Guide the model to break down complex reasoning into steps
   - Example: "Let's solve this step by step: {problem}"

4. Self-consistency:
   - Generate multiple responses and take the majority answer
   - Useful for reasoning and math problems

5. ReAct (Reasoning + Acting):
   - Alternate between reasoning and acting steps
   - Example: "Thought: I need to find X. Action: Search for X. Observation: Found Y about X."
"""

print("Example of effective prompt design:")
print('''
# Zero-shot prompt
zero_shot_prompt = """
Classify the following text into one of these categories: Business, Technology, Sports, Politics, Entertainment.

Text: {text}
Category:
"""

# Few-shot prompt
few_shot_prompt = """
Text: Apple announced their new iPhone model with improved camera features.
Category: Technology

Text: The senator proposed a new bill addressing climate change.
Category: Politics

Text: {text}
Category:
"""

# Chain of Thought prompt
cot_prompt = """
Question: If I have 5 apples and give 2 to my friend, then buy 3 more, how many apples do I have?

Let's think through this step by step:
1. Initially, I have 5 apples.
2. After giving 2 apples to my friend, I have 5 - 2 = 3 apples.
3. After buying 3 more apples, I have 3 + 3 = 6 apples.
Therefore, I have 6 apples.

Question: {question}

Let's think through this step by step:
"""
''')

# ===== HANDLING CONTEXT LIMITATIONS =====
print("\n===== HANDLING CONTEXT LIMITATIONS =====")
"""
LLMs have context length limitations. Here are strategies to handle them:

1. Chunking:
   - Split large documents into smaller chunks
   - Process each chunk separately or in a sliding window

2. Summarization:
   - Summarize long texts before processing
   - Use hierarchical summarization for very large documents

3. Retrieval-Augmented Generation (RAG):
   - Store document chunks in a vector database
   - Retrieve only relevant chunks when needed

4. Map-reduce approach:
   - Process chunks independently (map)
   - Combine the results (reduce)
"""

print("Example of handling large documents:")
print('''
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI

# Split a large document into chunks
with open("large_document.txt", "r") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = text_splitter.split_text(text)
print(f"Split into {len(chunks)} chunks")

# Method 1: Map-reduce summarization
llm = OpenAI(temperature=0)
map_reduce_chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = map_reduce_chain.run(chunks)
print(summary)

# Method 2: Retrieve relevant chunks based on a query
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(chunks, embeddings)

query = "What are the main points about climate change?"
relevant_chunks = db.similarity_search(query, k=3)
for chunk in relevant_chunks:
    print(chunk.page_content)
''')

# ===== EVALUATION AND TESTING =====
print("\n===== EVALUATION AND TESTING =====")
"""
Evaluating LLM applications is crucial for ensuring quality and reliability.

Evaluation Strategies:

1. Human evaluation:
   - Have humans rate outputs on relevance, accuracy, etc.
   - Use A/B testing to compare different approaches

2. Automated evaluation:
   - Use reference-based metrics (BLEU, ROUGE) for tasks with ground truth
   - Use LLM-as-a-judge to evaluate outputs
   - Create test cases with expected outputs

3. Behavioral testing:
   - Test invariance (output shouldn't change with irrelevant input changes)
   - Test directional expectations (how output should change with specific input changes)
   - Test for known failure modes and edge cases
"""

print("Example of LLM evaluation:")
print('''
from langchain.evaluation import load_evaluator
from langchain.llms import OpenAI

# Initialize evaluator
evaluator = load_evaluator("qa", llm=OpenAI(temperature=0))

# Evaluate a question-answer pair
eval_result = evaluator.evaluate_strings(
    prediction="Paris is the capital of France",
    reference="The capital of France is Paris",
    input="What is the capital of France?"
)
print(f"Score: {eval_result['score']}")
print(f"Reasoning: {eval_result['reasoning']}")

# Custom test cases
test_cases = [
    {
        "question": "What is 2+2?",
        "expected": "4"
    },
    {
        "question": "Who was the first president of the United States?",
        "expected": "George Washington"
    }
]

# Test an LLM chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])
chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

for test in test_cases:
    result = chain.run(question=test["question"])
    print(f"Question: {test['question']}")
    print(f"Expected: {test['expected']}")
    print(f"Actual: {result}")
    print(f"Pass: {'Yes' if test['expected'] in result else 'No'}")
    print()
''')

# ===== HANDLING HALLUCINATIONS =====
print("\n===== HANDLING HALLUCINATIONS =====")
"""
Hallucinations are when LLMs generate false or misleading information.

Strategies to Reduce Hallucinations:

1. Retrieval-Augmented Generation (RAG):
   - Ground responses in retrieved documents
   - Cite sources for information

2. Constrained generation:
   - Use structured outputs (JSON, XML)
   - Provide explicit instructions about what to do when uncertain

3. Self-verification:
   - Have the model verify its own outputs
   - Use multiple prompts to cross-check information

4. External verification:
   - Use search engines or knowledge bases to verify facts
   - Implement fact-checking mechanisms
"""

print("Example of reducing hallucinations:")
print('''
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Prompt that encourages the model to admit uncertainty
uncertainty_prompt = PromptTemplate(
    template="""
    Answer the following question based ONLY on the information provided. 
    If you're unsure or the information is not provided, say "I don't have enough information to answer this question."
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """,
    input_variables=["context", "question"]
)

llm = OpenAI(temperature=0)
response = llm(uncertainty_prompt.format(
    context="Paris is the capital of France. It has a population of about 2.2 million people.",
    question="What is the tallest building in Paris?"
))
print(response)

# Self-verification example
verification_prompt = PromptTemplate(
    template="""
    Question: {question}
    
    Initial answer: {initial_answer}
    
    Verify the initial answer. Check for factual errors or unsupported claims.
    If there are any issues, provide a corrected answer. If you're uncertain about any facts, acknowledge the uncertainty.
    
    Verified answer:
    """,
    input_variables=["question", "initial_answer"]
)

initial_answer = llm(f"Question: {question}\nAnswer:")
verified_answer = llm(verification_prompt.format(
    question=question,
    initial_answer=initial_answer
))
print(f"Initial answer: {initial_answer}")
print(f"Verified answer: {verified_answer}")
''')

# ===== BUILDING MULTI-AGENT SYSTEMS =====
print("\n===== BUILDING MULTI-AGENT SYSTEMS =====")
"""
Multi-agent systems involve multiple LLM-based agents working together to solve complex tasks.

Key Components:

1. Agent roles:
   - Define specialized roles for different agents
   - Examples: Researcher, Writer, Critic, Coordinator

2. Communication protocols:
   - Define how agents share information
   - Structured message formats

3. Coordination mechanisms:
   - How agents decide what to do next
   - Task allocation and sequencing

4. Memory and state management:
   - Shared knowledge base
   - Individual agent memories
"""

print("Example of a simple multi-agent system:")
print('''
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Define agent roles with system messages
researcher_system = """You are a Research Agent. Your job is to find and provide accurate information about a given topic.
Focus on facts and cite sources when possible. Ask clarifying questions if needed."""

writer_system = """You are a Writer Agent. Your job is to create well-written content based on information provided by the Research Agent.
Focus on clarity, engagement, and proper structure. Ask for more information if needed."""

editor_system = """You are an Editor Agent. Your job is to improve and refine the content created by the Writer Agent.
Focus on grammar, style, factual accuracy, and overall quality. Provide specific feedback and suggestions."""

# Initialize chat models for each agent
researcher = ChatOpenAI(temperature=0.2)
writer = ChatOpenAI(temperature=0.7)
editor = ChatOpenAI(temperature=0.2)

# Collaborative content creation process
def create_content(topic):
    # Step 1: Research phase
    research_messages = [
        SystemMessage(content=researcher_system),
        HumanMessage(content=f"I need information about {topic}. What are the key facts and details I should know?")
    ]
    research_response = researcher(research_messages)
    research_info = research_response.content
    print(f"RESEARCHER: {research_info}\\n")
    
    # Step 2: Writing phase
    writer_messages = [
        SystemMessage(content=writer_system),
        HumanMessage(content=f"Topic: {topic}\\nResearch information: {research_info}\\n\\nPlease write content about this topic.")
    ]
    writer_response = writer(writer_messages)
    draft_content = writer_response.content
    print(f"WRITER: {draft_content}\\n")
    
    # Step 3: Editing phase
    editor_messages = [
        SystemMessage(content=editor_system),
        HumanMessage(content=f"Please edit and improve the following content about {topic}:\\n\\n{draft_content}")
    ]
    editor_response = editor(editor_messages)
    final_content = editor_response.content
    print(f"EDITOR: {final_content}\\n")
    
    return final_content

# Use the multi-agent system
final_article = create_content("The impact of artificial intelligence on healthcare")
''')

# ===== FINE-TUNING LLMs =====
print("\n===== FINE-TUNING LLMs =====")
"""
Fine-tuning adapts pre-trained LLMs to specific tasks or domains.

When to Fine-tune:

1. Task specialization:
   - Improve performance on specific tasks
   - Examples: Medical diagnosis, legal document analysis

2. Style adaptation:
   - Match a specific writing style or tone
   - Examples: Brand voice, technical documentation

3. Knowledge integration:
   - Incorporate domain-specific knowledge
   - Examples: Company policies, product details

4. Instruction following:
   - Better adherence to specific formats or instructions
   - Examples: Structured outputs, consistent behavior

Fine-tuning Process:

1. Prepare training data (prompt-completion pairs)
2. Format data according to model requirements
3. Train the model (usually with lower learning rates)
4. Evaluate performance
5. Deploy the fine-tuned model
"""

print("Example of preparing data for fine-tuning:")
print('''
import json
import openai

# Example training data for a customer support assistant
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful customer support assistant for a software company."},
            {"role": "user", "content": "I can't log into my account."},
            {"role": "assistant", "content": "I'm sorry to hear you're having trouble logging in. Let's troubleshoot this together. Could you please tell me if you're getting any specific error message when trying to log in?"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful customer support assistant for a software company."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, please follow these steps:\\n1. Go to the login page\\n2. Click on 'Forgot Password'\\n3. Enter the email address associated with your account\\n4. Check your email for a password reset link\\n5. Click the link and follow the instructions to create a new password\\n\\nIf you don't receive the email within a few minutes, please check your spam folder. Let me know if you need any further assistance!"}
        ]
    }
]

# Save training data to a JSONL file
with open("training_data.jsonl", "w") as f:
    for entry in training_data:
        f.write(json.dumps(entry) + "\\n")

# Fine-tune the model (OpenAI example)
# openai.api_key = "your-api-key"
# response = openai.File.create(file=open("training_data.jsonl", "rb"), purpose="fine-tune")
# file_id = response.id
# fine_tune_job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
''')

# ===== DEPLOYMENT CONSIDERATIONS =====
print("\n===== DEPLOYMENT CONSIDERATIONS =====")
"""
Deploying LLM applications requires careful planning and infrastructure.

Key Considerations:

1. Hosting options:
   - Cloud providers (AWS, Azure, GCP)
   - Specialized LLM platforms (OpenAI, Anthropic, etc.)
   - Self-hosting (for open-source models)

2. Scalability:
   - Load balancing
   - Caching common requests
   - Asynchronous processing for long-running tasks

3. Monitoring:
   - Track usage and performance
   - Monitor for drift in quality
   - Log user interactions for improvement

4. Cost optimization:
   - Efficient prompt design
   - Caching responses
   - Using smaller models when appropriate

5. Security and privacy:
   - Data encryption
   - User data handling
   - Prompt injection prevention
"""

print("Example of a simple Flask API for an LLM application:")
print('''
from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

app = Flask(__name__)

# Initialize LLM
llm = OpenAI(temperature=0.7)

# Define prompt template
template = """
You are a helpful assistant.

User question: {question}

Your response:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Generate response
        formatted_prompt = prompt.format(question=question)
        response = llm(formatted_prompt)
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
''')

# ===== ETHICAL CONSIDERATIONS =====
print("\n===== ETHICAL CONSIDERATIONS =====")
"""
Building LLM applications comes with ethical responsibilities.

Key Ethical Considerations:

1. Bias and fairness:
   - Be aware of biases in training data
   - Test for and mitigate unfair outcomes
   - Consider diverse perspectives

2. Transparency:
   - Clearly indicate when users are interacting with an AI
   - Explain limitations and potential errors
   - Provide information about data usage

3. Privacy:
   - Minimize collection of personal data
   - Be transparent about data retention
   - Provide opt-out options

4. Safety:
   - Implement content filtering
   - Monitor for harmful outputs
   - Have human oversight for sensitive applications

5. Environmental impact:
   - Consider the computational resources used
   - Optimize for efficiency
   - Use smaller models when appropriate
"""

print("Example of implementing ethical guidelines:")
print('''
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Ethical guidelines embedded in the system prompt
ethical_prompt = PromptTemplate(
    template="""
    You are an AI assistant that follows these ethical guidelines:
    1. Provide balanced and fair responses that consider diverse perspectives
    2. Avoid generating harmful, illegal, unethical, or deceptive content
    3. Acknowledge limitations and uncertainties in your knowledge
    4. Respect user privacy and confidentiality
    5. Provide sources or clarify when information might not be reliable
    
    User query: {query}
    
    Your response:
    """,
    input_variables=["query"]
)

llm = OpenAI(temperature=0.7)
response = llm(ethical_prompt.format(query="Tell me about immigration policies"))
print(response)

# Content moderation example
def moderate_content(text):
    """Simple content moderation function"""
    # In a real application, you would use a more sophisticated approach
    # or a dedicated content moderation API
    problematic_terms = ["hate", "violence", "illegal", "harm"]
    for term in problematic_terms:
        if term in text.lower():
            return True, f"Content contains problematic term: {term}"
    return False, "Content passed moderation"

user_input = "How to make a bomb"
is_problematic, reason = moderate_content(user_input)
if is_problematic:
    print(f"Input rejected: {reason}")
    response = "I cannot provide information on harmful or illegal activities."
else:
    response = llm(ethical_prompt.format(query=user_input))
    print(response)
''')

# ===== CONCLUSION =====
print("\n===== CONCLUSION =====")
print("""
Building effective LLM applications requires:
1. Thoughtful prompt engineering
2. Strategies for handling context limitations
3. Robust evaluation and testing
4. Techniques to reduce hallucinations
5. Consideration of deployment requirements
6. Attention to ethical implications

As LLM technology continues to evolve, these best practices will help you build applications that are:
- Reliable and accurate
- Scalable and efficient
- Ethical and responsible
- Valuable to users

Remember that the field is rapidly changing, so stay updated with the latest research and techniques.
""")

print("\n===== END OF BUILDING LLM APPLICATIONS =====")