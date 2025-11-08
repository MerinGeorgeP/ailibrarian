📚 AI Librarian
Your Personal AI-Powered Document Assistant

AI Librarian is a full-stack application that helps users upload, search, summarize, and manage their PDF documents using advanced Natural Language Processing (NLP) and AI models.
It combines a FastAPI backend with a Streamlit frontend, providing a seamless and interactive experience.

🚀 Features

✅ User Authentication
Secure signup and login using JWT tokens.
Each user has a private workspace and their own FAISS index.

✅ PDF Upload & Storage
Upload PDFs directly through the app.
Files and metadata are stored per user in organized folders.

✅ Semantic Search
Search through uploaded PDFs using Sentence Transformers for vector embeddings.
Finds the most relevant documents based on meaning, not just keywords.

✅ Summarization
Summarize long documents with Facebook BART Large CNN, creating concise overviews.

✅ File Management
View, download, and delete PDFs from your library.
Automatically rebuilds FAISS index after deletions.

✅ Frontend
Built with Streamlit for an intuitive and responsive user interface.

🧩 Tech Stack
Layer	Technology: Backend	FastAPI
Frontend: Streamlit
Embedding Model:	all-MiniLM-L6-v2 (Sentence Transformers)
Summarizer:	facebook/bart-large-cnn
Vector Indexing:	FAISS
Authentication:	JWT + Password Hashing (Passlib)
PDF Processing:	PyPDF2

🧱 Project Structure
AI-Librarian/
│
├── backend.py          # FastAPI backend
├── frontend.py         # Streamlit frontend
├── data/               # Stores PDFs, metadata, and FAISS indexes
│   ├── pdfs/
│   ├── meta.json
│   ├── users/
│   └── index.faiss
└── requirements.txt
