Academic Research Assistant 🎓🔍
A Flask-based web application that helps researchers find, analyze, and summarize academic documents using AI-powered search and natural language processing.

🔑 Key Features
Smart Document Search: Finds relevant PDFs, PPTs, and Word documents using SerpAPI

Metadata Extraction: Automatically detects document titles, authors, and publication years

Semantic Analysis: Uses sentence transformers to find the most relevant content

AI Summarization: Generates comprehensive summaries using Llama3

Document Classification: Organizes results by document type (PDFs, PPTs, etc.)

🛠️ Technology Stack
Backend: Python Flask

Search API: SerpAPI (Google Search)

NLP Models:

all-MiniLM-L6-v2 (Sentence Transformers) for semantic search

Llama3 (via Ollama) for summarization and metadata extraction

Document Processing: PyPDF2 for text extraction

Frontend: HTML/CSS/JavaScript (not included in this repo)

⚙️ How It Works
🔍 Search Phase
User enters research query and selects document types

System searches academic sources using SerpAPI

Extracts metadata from URLs and document headers

📊 Analysis Phase
Downloads selected documents

Extracts and chunks text content

Finds most relevant sections using cosine similarity

Generates comprehensive summaries using Llama3

✅ Results
Document metadata (title, author, source)

AI-generated summary

Relevance scores

🚀 Setup Instructions
📦 Prerequisites
Python 3.8+

Ollama with Llama3 installed

bash
Copy
Edit
ollama pull llama3
SerpAPI account

🔧 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/academic-research-assistant.git
cd academic-research-assistant
pip install -r requirements.txt
🔐 Configuration
Create a .env file in the root directory and add your API key:

ini
Copy
Edit
SERPAPI_API_KEY=your_serpapi_key
▶️ Running the Application
Start Ollama server in a separate terminal:

bash
Copy
Edit
ollama serve
Then run the Flask application:

bash
Copy
Edit
python app.py
🌐 API Endpoints
POST /search
Search for academic documents
Parameters:

query (string)

types (array of document types)

POST /analyze
Analyze a specific document
Parameters:

url (string)

query (string)

📖 Example Usage
python
Copy
Edit
# Search for machine learning papers
response = requests.post("http://localhost:5000/search", json={
    "query": "machine learning in healthcare",
    "types": ["PDF", "Research"]
})

# Analyze a specific document
response = requests.post("http://localhost:5000/analyze", json={
    "url": "https://arxiv.org/pdf/2103.12345.pdf",
    "query": "machine learning applications in radiology"
})
🔮 Future Enhancements
Add support for more document types (ePub, HTML)

Implement citation extraction

Add reference management integration

Develop browser extension version

🤝 Contributing
Contributions are welcome!
Feel free to open an issue or submit a pull request.
