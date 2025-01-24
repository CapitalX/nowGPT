from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime
import json
from pathlib import Path
import asyncio  # Add this import at the top
import PyPDF2
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def display_sample_chunks(chunks, num_samples=10):
    print("\n=== Sample Chunks and Metadata ===")
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\nChunk {i + 1}:")
        print("Content:", chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)
        print("Metadata:", chunk.metadata)
        print("-" * 80)

def process_single_pdf(pdf_path, filename):
    try:
        print(f"\nStarting to process: {filename}")
        
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB
        print(f"File size: {file_size:.2f} MB")
        
        if file_size > 100:
            print(f"Warning: {filename} is quite large ({file_size:.2f} MB). Processing may take a while...")
        
        # Use a more robust PDF loading approach
        try:
            loader = PyPDFLoader(pdf_path)
            print(f"Loading pages from: {filename}")
            doc_pages = []
            
            # Load pages one at a time with error handling
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        doc_pages.append(Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "total_pages": total_pages
                            }
                        ))
                        if (page_num + 1) % 10 == 0:  # Progress update every 10 pages
                            print(f"Processed {page_num + 1} of {total_pages} pages")
                    except Exception as e:
                        print(f"Error on page {page_num + 1}: {str(e)}")
                        continue
                        
            print(f"Successfully loaded {len(doc_pages)} pages from {filename}")
            
            if not doc_pages:
                print("No pages were successfully loaded!")
                return 0
                
            print("\nSplitting document into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            # Process chunks in smaller batches
            batch_size = 100
            all_chunks = []
            for i in range(0, len(doc_pages), batch_size):
                batch = doc_pages[i:i + batch_size]
                chunks = text_splitter.split_documents(batch)
                all_chunks.extend(chunks)
                print(f"Processed chunks for pages {i} to {min(i + batch_size, len(doc_pages))} of {len(doc_pages)}")
            
            print(f"\nCreated {len(all_chunks)} total chunks")
            display_sample_chunks(all_chunks[:3])  # Show first 3 chunks
            
            print("\nCreating vector store in Supabase...")
            # Process in smaller batches for vector store creation
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                vector_store = SupabaseVectorStore.from_documents(
                    batch,
                    embeddings,
                    client=supabase,
                    table_name="documents",
                    query_name="match_documents"
                )
                print(f"Vectorized chunks {i} to {min(i + batch_size, len(all_chunks))} of {len(all_chunks)}")
            
            print(f"Successfully vectorized {filename}")
            return len(all_chunks)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            return 0
            
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return 0

def get_processed_files():
    processed_files_path = Path("processed_files.json")
    if processed_files_path.exists():
        with open(processed_files_path, "r") as f:
            return json.load(f)
    return {"processed": []}

def mark_file_as_processed(filename):
    processed_files_path = Path("processed_files.json")
    processed = get_processed_files()
    processed["processed"].append(filename)
    with open(processed_files_path, "w") as f:
        json.dump(processed, f, indent=2)

async def check_file_in_supabase(filename):
    try:
        # Simplified query without filter parameter
        response = supabase.rpc(
            'match_documents',
            {
                'query_embedding': embeddings.embed_query(filename),  # Use filename as query
                'match_count': 1
            }
        ).execute()  # Remove await here since execute() returns a regular response
        
        # Check if any results have matching filename in metadata
        if response.data:
            for doc in response.data:
                if doc.get('metadata', {}).get('source') == filename:
                    return True
        return False
    except Exception as e:
        print(f"Error checking Supabase for {filename}: {str(e)}")
        return False

async def process_pdfs(pdf_directory):
    total_chunks = 0
    
    print(f"Scanning directory: {pdf_directory}")
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    print(f"Found PDF files: {pdf_files}")
    
    # Get list of already processed files from local JSON
    processed_files = get_processed_files()["processed"]
    print(f"\nPreviously processed files (local): {processed_files}")
    
    # Check Supabase for each file
    files_to_process = []
    for filename in pdf_files:
        if filename in processed_files:
            print(f"Skipping {filename} (found in local processed files)")
            continue
            
        exists_in_supabase = await check_file_in_supabase(filename)
        if exists_in_supabase:
            print(f"Skipping {filename} (found in Supabase)")
            # Add to local processed files if not already there
            if filename not in processed_files:
                mark_file_as_processed(filename)
            continue
            
        files_to_process.append(filename)
    
    print(f"\nFiles remaining to process: {files_to_process}")
    
    if not files_to_process:
        print("No new files to process!")
        return 0
    
    for filename in files_to_process:
        pdf_path = os.path.join(pdf_directory, filename)
        chunks_processed = process_single_pdf(pdf_path, filename)
        if chunks_processed > 0:
            mark_file_as_processed(filename)
            total_chunks += chunks_processed
            print(f"Completed processing {filename}. Total chunks so far: {total_chunks}\n")
        print("-" * 80 + "\n")
    
    return total_chunks

if __name__ == "__main__":
    pdf_directory = "/Users/itscapitalx/Desktop/Xanadu PDFS"
    
    print(f"Starting PDF processing...")
    
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' does not exist")
        exit(1)
    
    # Run the async function
    num_chunks = asyncio.run(process_pdfs(pdf_directory))
    print(f"Finished processing all PDFs. Total chunks processed: {num_chunks}") 