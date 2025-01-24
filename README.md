# AI Documentation Chat Assistant

An AI-powered chat interface for querying any documentation using RAG (Retrieval Augmented Generation) technology. This tool allows you to create an intelligent assistant that can answer questions about your documentation with source references and context.

## Features

- Interactive chat interface for your documentation
- PDF processing and vectorization
- Document statistics and management
- Source reference tracking
- Conversation history
- Rich terminal interface

## Prerequisites

- Python 3.8+
- OpenAI API key
- Supabase account and project
- Your PDF documentation files

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-doc-assistant
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

## Setup

1. Create a Supabase project and create the vector store table:
```sql
create extension if not exists vector;

create table documents (
    id uuid primary key default uuid_generate_v4(),
    content text,
    metadata jsonb,
    embedding vector(1536)
);

create index on documents using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
```

2. Update the PDF directory path in `pdf_processor.py`:
```python
pdf_directory = "/path/to/your/pdfs"  # Change to your PDF directory
```

3. Process your PDFs:
```bash
python pdf_processor.py
```

## Usage

1. Start the chat interface:
```bash
python chat.py
```

2. Available commands:
- Type your questions normally for answers
- `exit` - End the chat session
- `clear` - Reset conversation history
- `sources` - Toggle source display
- `refresh` - Update document statistics

## Customization

### 1. Prompt Engineering
Modify the assistant's personality and response format in `chat.py`:
```python
prompt_template = """You are an expert assistant for [YOUR DOMAIN].
...
Please provide your response in the following format:
1. Direct Answer: [Concise answer to the question]
2. Additional Details: [Relevant supporting information]
3. Related Topics: [Suggest 2-3 related topics]
"""
```

### 2. Document Processing
Adjust chunking parameters in `pdf_processor.py` for your documentation:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust based on your content
    chunk_overlap=200,  # Adjust for context preservation
)
```

### 3. Model Settings
Fine-tune the AI response in `chat.py`:
```python
llm=ChatOpenAI(
    temperature=0.7,  # Adjust for creativity (0.0) vs accuracy (1.0)
)
retriever=vector_store.as_retriever(
    search_kwargs={"k": 4}  # Number of chunks to retrieve
)
```

## Performance Tips

1. **Large Documents**
   - Increase batch size for faster processing
   - Adjust chunk size for better context
   - Monitor memory usage

2. **Response Quality**
   - Tune temperature for desired creativity
   - Adjust chunk overlap for context
   - Modify number of retrieved chunks

3. **Memory Usage**
   - Use the refresh command sparingly
   - Clear chat history for long sessions
   - Monitor system resources

## Troubleshooting

Common issues and solutions:
1. **PDF Processing Errors**
   - Check file permissions
   - Verify PDF is not corrupted
   - Ensure sufficient disk space

2. **Vector Store Issues**
   - Verify Supabase connection
   - Check table creation
   - Monitor API rate limits

3. **Memory Problems**
   - Reduce batch size
   - Clear chat history
   - Process fewer documents at once

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

