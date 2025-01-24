# ServiceNow Xanadu Documentation Chat Assistant

An AI-powered chat interface for querying ServiceNow Xanadu documentation using RAG (Retrieval Augmented Generation) technology.

## Features

- Interactive chat interface with the Xanadu documentation
- PDF processing and vectorization
- Document statistics and management
- Source reference tracking
- Conversation history
- Rich terminal interface

## Prerequisites

- Python 3.8+
- OpenAI API key
- Supabase account and project
- ServiceNow Xanadu PDF documentation files

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd xanadu-chat-assistant
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

1. Create the Supabase vector store table:
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

2. Place your Xanadu PDF documentation files in the designated directory:
```
/Users/itscapitalx/Desktop/Xanadu PDFS/
```

3. Process the PDFs (only needed once):
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

## Features in Detail

### Document Processing
- Automatic PDF text extraction
- Chunking for optimal retrieval
- Vector embedding generation
- Progress tracking for processed files

### Chat Interface
- Rich text formatting
- Loading indicators
- Source document references
- Conversation memory
- Error handling

### Document Statistics
- File sizes
- Page counts
- Chunk counts
- Content previews
- Cached statistics for quick startup

## Response Format

The AI assistant provides responses in a structured format:
1. Direct Answer - Concise response to the question
2. Additional Details - Supporting information
3. Related Topics - Suggested related areas to explore

## Maintenance

- Use the `refresh` command to update document statistics
- Monitor the `processed_files.json` for tracking processed documents
- Check `document_stats.json` for cached statistics

## Files

- `chat.py` - Main chat interface
- `pdf_processor.py` - PDF processing utility
- `processed_files.json` - Tracks processed documents
- `document_stats.json` - Cached document statistics

## Error Handling

The tool includes robust error handling for:
- PDF processing issues
- Vector store operations
- Chat interactions
- File operations

## Performance

- Cached document statistics for fast startup
- Batch processing for large PDFs
- Optimized vector search
- Memory management for long conversations

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License

Copyright (c) 2024 XTech Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Configuration and Customization

### Essential Configuration

1. **Environment Variables** (.env file):
   ```env
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```

2. **PDF Directory Path** (pdf_processor.py):
   ```python
   pdf_directory = "/Users/itscapitalx/Desktop/Xanadu PDFS"  # Change to your PDF directory
   ```
   Also update this path in chat.py's get_document_stats function.

### AI Model Customization

1. **Chunk Size** (pdf_processor.py):
   ```python
   text_splitter = RecursativeCharacterTextSplitter(
       chunk_size=1000,  # Adjust for different document lengths
       chunk_overlap=200,  # Adjust for context preservation
   )
   ```

2. **LLM Parameters** (chat.py):
   ```python
   llm=ChatOpenAI(
       temperature=0.7,  # Adjust for creativity vs accuracy (0.0-1.0)
   )
   ```

3. **Context Window** (chat.py):
   ```python
   retriever=vector_store.as_retriever(
       search_kwargs={"k": 4}  # Adjust number of chunks retrieved
   )
   ```

### Prompt Engineering

1. **Assistant Personality** (chat.py):
   ```python
   prompt_template = """You are a ServiceNow Xanadu expert assistant...
   // Customize the prompt template for different use cases
   """
   ```

2. **Response Format** (chat.py):
   Modify the response structure in the prompt template:
   ```python
   Please provide your response in the following format:
   1. Direct Answer: [Concise answer to the question]
   2. Additional Details: [Relevant supporting information]
   3. Related Topics: [Suggest 2-3 related topics]
   ```

### Performance Optimization

1. **Batch Processing** (pdf_processor.py):
   ```python
   batch_size = 100  # Adjust based on available memory
   ```

2. **Vector Search** (supabase_setup.sql):
   ```sql
   create index on documents using ivfflat (embedding vector_cosine_ops)
       with (lists = 100);  # Adjust for dataset size
   ```

### UI Customization

1. **Console Styling** (chat.py):
   ```python
   # Modify Rich console styles
   border_style="cyan"  # Change colors
   style="dim"         # Adjust text styles
   ```

2. **Display Options** (chat.py):
   - Modify table columns in display_document_stats
   - Adjust preview lengths in display_sources
   - Customize progress bar messages

### Error Handling

1. **Chunk Size Limits** (pdf_processor.py):
   ```python
   if file_size > 100:  # Adjust warning threshold for large files
       print(f"Warning: {filename} is quite large...")
   ```

2. **Retry Logic** (chat.py):
   - Add custom retry logic for API calls
   - Modify error messages and recovery behavior

Remember to test thoroughly after making any modifications, especially when adjusting chunk sizes or batch processing parameters, as these can impact both performance and memory usage.