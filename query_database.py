from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Configuration
CHROMA_PATH = 'chroma'
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"
CHAT_MODEL = "models/gemini-1.5-flash-001"

def initialize_embeddings():
    """Initialize Google Generative AI embeddings."""
    
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        task_type=EMBEDDING_TASK_TYPE
    )

def load_chroma_database(embedding_function):
    """Load Chroma database from the specified path."""
    
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def create_prompt_template():
    """Create the prompt template for generating responses."""
    
    return PromptTemplate.from_template(''' 
        Answer the question as detailed as possible based on the provided context. Make sure to provide all the details. If the answer is not in the provided context just say, "No answer available." Don't give a wrong answer.

        Context:
        {context}

        Question:
        {question}

        Answer:
    ''')

def retrieve_context_from_db(db, question, k=1):
    """Retrieve context from the database based on the user's question."""
    results = db.similarity_search_with_relevance_scores(question, k=k)
    
    return "\n".join([result.page_content for result, score in results])

def generate_answer(model, prompt):
    """Generate an answer using the ChatGoogleGenerativeAI model."""
    human_message = HumanMessage(content=prompt)
    response = model.invoke([human_message])
    
    return response.content

def main():
    # Initialize components
    embeddings = initialize_embeddings()
    db = load_chroma_database(embedding_function=embeddings)
    prompt_template = create_prompt_template()
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    
    # Query
    question = input("Enter your question: ")
    
    # Process query
    context = retrieve_context_from_db(db, question)
    formatted_prompt = prompt_template.format(context=context, question=question)
    answer = generate_answer(model, formatted_prompt)
    
    # Print the answer
    print(f'Answer: {answer}')

if __name__ == '__main__':
    main()
