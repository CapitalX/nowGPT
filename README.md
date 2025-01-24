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