# Academic Research Assistant 🎓🔍

A **Flask-based web application** that helps researchers find, analyze, and summarize academic documents using **AI-powered search** and **natural language processing**.

---

## ✨ Key Features

- **Smart Document Search**: Finds relevant PDFs, PPTs, and Word documents using SerpAPI  
- **Metadata Extraction**: Automatically detects document titles, authors, and publication years  
- **Semantic Analysis**: Uses sentence transformers to find the most relevant content  
- **AI Summarization**: Generates comprehensive summaries using Llama3  
- **Document Classification**: Organizes results by document type (PDFs, PPTs, etc.)

---

## 🛠️ Technology Stack

- **Backend**: Python Flask  
- **Search API**: SerpAPI (Google Search)  
- **NLP Models**:
  - `all-MiniLM-L6-v2` (Sentence Transformers) for semantic search  
  - `Llama3` (via Ollama) for summarization and metadata extraction  
- **Document Processing**: PyPDF2 for text extraction  
- **Frontend**: HTML/CSS/JavaScript *(not included in this repo)*

