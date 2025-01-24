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
from langchain.schema import BaseRetriever, Document
from pydantic import Field, BaseModel
from langchain.memory import ConversationBufferWindowMemory

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
        progress.update(task1, completed=100)
        
        task2 = progress.add_task("[green]Setting up Vector Store...", total=100)
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        progress.update(task2, completed=100)

        @tool
        def select_relevant_sources(query: str) -> List[str]:
            """
            Analyzes the query to determine which documentation sources are most relevant.
            Returns a list of relevant PDF filenames.
            """
            sources_descriptions = {
                "xanadu_platform_security.pdf": "Security features, authentication methods, authorization controls, compliance standards, data protection",
                "xanadu_general_release_notes.pdf": "Product updates, new feature releases, improvements, bug fixes, deprecation notices",
                "xanadu_api_references.pdf": "Complete API documentation, endpoint specifications, request/response formats, authentication methods",
                "xanadu_application_development.pdf": "Development guidelines, scripting tutorials, customization options, best practices",
                "xanadu_it_service_management.pdf": "ITSM workflows, incident/problem management, service desk operations, SLA management",
                "xanadu_glossary.pdf": "Comprehensive technical terms, industry definitions, platform-specific concepts",
                "xanadu_customer_service_management.pdf": "Customer service features, case management workflows, SLA tracking, customer engagement tools"
            }
            
            source_selector = ChatOpenAI(temperature=0)
            response = source_selector.invoke(
                f"""Given this query: '{query}'
                Select the most relevant documentation sources from:
                {json.dumps(sources_descriptions, indent=2)}
                
                Return only the filenames in a comma-separated list.
                Consider:
                1. Query topic and intent
                2. Technical vs business focus
                3. Specific feature mentions
                
                Return format: filename1.pdf,filename2.pdf
                """
            )
            
            # Extract the content from the response
            if hasattr(response, 'content'):
                return response.content.strip().split(',')
            return response.strip().split(',')

        class SmartRetriever(BaseRetriever):
            """Custom retriever that filters by source before searching."""
            
            vectorstore: SupabaseVectorStore = Field(description="Vector store for document retrieval")
            relevant_sources: List[str] = Field(default_factory=list, description="Currently selected source documents")

            class Config:
                arbitrary_types_allowed = True

            def __init__(self, vectorstore: SupabaseVectorStore):
                super().__init__(vectorstore=vectorstore)
                self.vectorstore = vectorstore
                self.relevant_sources = []

            def _get_relevant_documents(self, query: str) -> List[Document]:
                """Get documents relevant to a query."""
                tool_response = select_relevant_sources.invoke(query)
                self.relevant_sources = tool_response.strip().split(',') if isinstance(tool_response, str) else tool_response
                console.print(f"[dim]Searching in: {', '.join(self.relevant_sources)}[/dim]")
                
                filter_dict = {
                    "sources": {
                        "$in": self.relevant_sources
                    }
                }
                
                docs = self.vectorstore.similarity_search(
                    query,
                    k=4,
                    filter=filter_dict
                )
                
                # Ensure each document has the required metadata
                for doc in docs:
                    if 'sources' not in doc.metadata:
                        doc.metadata['sources'] = doc.metadata.get('source', 'Unknown')
                
                return docs
                
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                """Async version of get_relevant_documents."""
                return self._get_relevant_documents(query)

        prompt_template = """You are an expert assistant in ServiceNow Xanadu Release Notes and Documentation. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}

        Chat History:
        {chat_history}

        Question: {question}

        Please provide your response in the following format:

        1. Direct Answer: 
        [Provide a clear, concise answer directly addressing the question]

        2. From the Documentation:
        [Quote or summarize relevant information from the provided context]

        3. Additional Details:
        [Provide any important context, examples, code snippets, or clarifications]

        4. Related Documentation:
        [List 2-3 related topics from the Xanadu documentation that might be helpful]

        Format your response using markdown for better readability.
        Use bullet points and code blocks where appropriate.
        If showing technical steps, number them clearly.

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "chat_history", "question"]
        )
        
        task3 = progress.add_task("[yellow]Creating QA Chain...", total=100)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7),
            retriever=SmartRetriever(vectorstore=vector_store),
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PROMPT,
                "document_variable_name": "context",
                "document_separator": "\n\n"
            },
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                k=3
            ),
            verbose=True
        )
        progress.update(task3, completed=100)
        
    return qa_chain, vector_store

def display_welcome():
    welcome_text = """
    # ServiceNow Xanadu Release Chat Assistant
    
    Ask questions about the Xanadu release documentation.
    - Type 'exit' to end the conversation
    - Type 'clear' to clear chat history
    - Type 'sources' to toggle sources display
    """
    console.print(Panel(Markdown(welcome_text), border_style="cyan"))

def display_source(sources):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("sources", style="dim")
    table.add_column("Page", justify="right")
    table.add_column("Preview", style="cyan")
    
    for doc in sources:
        table.add_row(
            doc.metadata['sources'],
            str(doc.metadata.get('page', 'N/A')),
            doc.page_content[:100] + "..."
        )
    
    console.print(Panel(table, title="sources", border_style="blue"))

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
                chunk_count = len([r for r in results if r.metadata['sources'] == filename])
                
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
    
    qa_chain, vector_store = initialize_qa_chain()
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
                console.print(f"[yellow]Sources display {'enabled' if show_sources else 'disabled'}[/]")
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
            if show_sources and "source_documents" in result:
                display_source(result["source_documents"])
            
            chat_history.append((question, result["answer"]))
            
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/]")
            console.print("[yellow]Please try asking your question again.[/]")

if __name__ == "__main__":
    chat() 