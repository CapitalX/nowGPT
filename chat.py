from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from supabase import create_client
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress
from rich.table import Table
from rich import print as rprint
import time
import json
import PyPDF2
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import tool
from typing import List, Dict

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

def initialize_qa_chain():
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Initializing OpenAI...", total=100)
        embeddings = OpenAIEmbeddings()
        progress.update(task1, advance=30)
        
        task2 = progress.add_task("[green]Setting up Vector Store...", total=100)
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        progress.update(task2, advance=100)

        # Define source selection tool
        @tool
        def select_relevant_sources(query: str) -> List[str]:
            """
            Analyzes the query to determine which documentation sources are most relevant.
            Returns a list of relevant PDF filenames.
            """
            source_descriptions = {
                "xanadu_platform_security.pdf": "Security, authentication, authorization, compliance",
                "xanadu_general_release_notes.pdf": "Release updates, changes, features, bug fixes",
                "xanadu_api_references.pdf": "API documentation, endpoints, methods, parameters",
                "xanadu_application_development.pdf": "Development guides, scripting, customization",
                "xanadu_it_service_management.pdf": "ITSM processes, incident management, service desk",
                "xanadu_glossary.pdf": "Technical terms, definitions, concepts",
                "xanadu_customer_service_management.pdf": "CSM features, case management, customer service"
            }
            
            # Create an agent to select sources
            source_selector = ChatOpenAI(temperature=0)
            response = source_selector.predict(
                f"""Given this query: '{query}'
                Select the most relevant documentation sources from:
                {json.dumps(source_descriptions, indent=2)}
                
                Return only the filenames in a comma-separated list.
                Consider:
                1. Query topic and intent
                2. Technical vs business focus
                3. Specific feature mentions
                
                Return format: filename1.pdf,filename2.pdf
                """
            )
            
            return response.strip().split(',')

        # Enhance retriever with filtered search
        def filtered_retriever(query: str):
            relevant_sources = select_relevant_sources(query)
            console.print(f"[dim]Searching in: {', '.join(relevant_sources)}[/dim]")
            
            # Create metadata filter
            filter_dict = {
                "source": {
                    "$in": relevant_sources
                }
            }
            
            # Return filtered search results
            return vector_store.similarity_search_with_score(
                query,
                k=4,
                filter=filter_dict
            )

        # Custom prompt template with source context
        prompt_template = """You are an expert assistant. The user's question appears to be related to documentation from: {sources}.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}

        Chat History:
        {chat_history}

        Question: {question}

        Please provide your response in the following format:
        1. Direct Answer: [Concise answer to the question]
        2. Additional Details: [Relevant supporting information]
        3. Related Topics: [Suggest 2-3 related topics]
        4. Source Context: [Brief explanation of why these sources were chosen]

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "chat_history", "question", "sources"]
        )
        
        task3 = progress.add_task("[yellow]Creating QA Chain...", total=100)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7),
            retriever=filtered_retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            ),
            verbose=True
        )
        progress.update(task3, advance=100)
        
    return qa_chain, vector_store

def display_welcome():
    welcome_text = """
    # ServiceNow Xanadu Release Chat Assistant
    
    Ask questions about the Xanadu release documentation.
    - Type 'exit' to end the conversation
    - Type 'clear' to clear chat history
    - Type 'sources' to toggle source display
    """
    console.print(Panel(Markdown(welcome_text), border_style="cyan"))

def display_sources(sources):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Source", style="dim")
    table.add_column("Page", justify="right")
    table.add_column("Preview", style="cyan")
    
    for doc in sources:
        table.add_row(
            doc.metadata['source'],
            str(doc.metadata.get('page', 'N/A')),
            doc.page_content[:100] + "..."
        )
    
    console.print(Panel(table, title="Sources", border_style="blue"))

def load_cached_stats():
    cache_file = 'document_stats.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def save_stats_cache(stats):
    with open('document_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

def get_document_stats(vector_store, force_refresh=False):
    if not force_refresh:
        cached_stats = load_cached_stats()
        if cached_stats:
            return cached_stats

    stats = {
        "documents": [],
        "totals": {
            "pages": 0,
            "size": 0,
            "chunks": 0
        },
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open('processed_files.json', 'r') as f:
            processed = json.load(f)
        
        for filename in processed['processed']:
            try:
                file_path = os.path.join("/Users/itscapitalx/Desktop/Xanadu PDFS", filename)
                size_bytes = os.path.getsize(file_path)
                size_mb = size_bytes / (1024 * 1024)
                
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    page_count = len(pdf_reader.pages)
                    preview = pdf_reader.pages[0].extract_text()[:100].replace('\n', ' ')
                
                results = vector_store.similarity_search(
                    f"filename:{filename}",
                    k=1000
                )
                chunk_count = len([r for r in results if r.metadata['source'] == filename])
                
                doc_stats = {
                    "filename": filename,
                    "pages": page_count,
                    "size_mb": round(size_mb, 1),
                    "chunks": chunk_count,
                    "preview": f"{preview}..."
                }
                
                stats["documents"].append(doc_stats)
                stats["totals"]["pages"] += page_count
                stats["totals"]["size"] += size_mb
                stats["totals"]["chunks"] += chunk_count
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        save_stats_cache(stats)
        return stats
        
    except Exception as e:
        print(f"Error gathering stats: {str(e)}")
        return None

def display_document_stats(stats):
    if not stats:
        console.print("[red]No document statistics available")
        return
        
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Document", style="dim")
    table.add_column("Pages", justify="right")
    table.add_column("File Size", justify="right", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    table.add_column("Preview", style="yellow")
    
    for doc in stats["documents"]:
        table.add_row(
            doc["filename"],
            str(doc["pages"]),
            f"{doc['size_mb']:.1f} MB",
            str(doc["chunks"]),
            doc["preview"]
        )
    
    # Add totals row
    table.add_row(
        "[bold]Total",
        f"[bold]{stats['totals']['pages']}",
        f"[bold]{stats['totals']['size']:.1f} MB",
        f"[bold]{stats['totals']['chunks']}",
        ""
    )
    
    console.print(Panel(
        table,
        title=f"Document Store Summary (Last Updated: {stats['last_updated']})",
        border_style="green"
    ))

def chat():
    display_welcome()
    
    # Initialize QA chain with progress bar
    qa_chain, vector_store = initialize_qa_chain()
    
    # Load cached stats instead of checking files every time
    stats = get_document_stats(vector_store)
    display_document_stats(stats)
    
    chat_history = []
    show_sources = True
    
    while True:
        try:
            question = console.input("\n[bold green]Your question: [/]")
            
            if question.lower() == 'exit':
                console.print("\n[yellow]Thanks for using the Xanadu Chat Assistant![/]")
                break
                
            if question.lower() == 'clear':
                chat_history = []
                console.print("[yellow]Chat history cleared![/]")
                continue
                
            if question.lower() == 'sources':
                show_sources = not show_sources
                console.print(f"[yellow]Source display {'enabled' if show_sources else 'disabled'}[/]")
                continue
                
            if question.lower() == 'refresh':
                console.print("[yellow]Refreshing document statistics...[/]")
                stats = get_document_stats(vector_store, force_refresh=True)
                display_document_stats(stats)
                continue
            
            with console.status("[bold green]Thinking..."):
                result = qa_chain.invoke({
                    "question": question, 
                    "chat_history": chat_history
                })
            
            # Display answer in a panel
            console.print(Panel(
                result["answer"],
                title="Answer",
                border_style="green"
            ))
            
            # Display sources if enabled
            if show_sources:
                display_sources(result["source_documents"])
            
            chat_history.append((question, result["answer"]))
            
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/]")
            console.print("[yellow]Please try asking your question again.[/]")

if __name__ == "__main__":
    chat() 