import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from src import config

def format_docs(retrieved_docs):
    """Formats retrieved Document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def get_llm() -> ChatGroq:
    """Initializes and returns the ChatGroq model using the API key from environment."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Please make sure your .env file is set up correctly.")
    return ChatGroq(
        model=config.GROQ_MODEL_NAME,
        temperature=0.2,
        groq_api_key=api_key
    )

def create_rag_chain(retriever):
    """
    Creates and returns the LCEL RAG chain.
    The chain takes a string question and returns a dictionary:
    {
        "context": list[Document],
        "question": str,
        "answer": str
    }
    """
    llm = get_llm()
    parser = StrOutputParser()
    
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}

Answer:
""",
        input_variables=['context', 'question']
    )
    
    # Chain to format docs and call LLM
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | parser
    )
    
    # Main parallel chain to fetch docs and pipe to generator
    main_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return main_chain
