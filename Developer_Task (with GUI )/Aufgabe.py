from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
import threading
import time
from datetime import datetime
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# RAG Pipeline Class (embedded)
class RAGPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.chunks = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]

    def process_pdfs(self, pdf_directory: str, progress_callback=None):
        if progress_callback:
            progress_callback("Starting PDF processing...")

        self.documents = []
        self.chunks = []

        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]

        for i, filename in enumerate(pdf_files):
            if progress_callback:
                progress_callback(f"Processing {filename} ({i+1}/{len(pdf_files)})")

            pdf_path = os.path.join(pdf_directory, filename)
            text = self.extract_text_from_pdf(pdf_path)

            if text:
                chunks = self.chunk_text(text)
                for chunk in chunks:
                    self.documents.append({
                        'filename': filename,
                        'text': chunk
                    })
                    self.chunks.append(chunk)

        if progress_callback:
            progress_callback(f"Creating embeddings for {len(self.chunks)} chunks...")

        embeddings = self.model.encode(self.chunks, show_progress_bar=False)

        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        if progress_callback:
            progress_callback("Processing complete!")

        return len(pdf_files), len(self.chunks)

    def search_similar(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        if self.index is None:
            raise ValueError("Index not created. Please process PDFs first.")

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx]['text'],
                    float(score),
                    self.documents[idx]['filename']
                ))

        return results

    def answer_question(self, question: str) -> Dict:
        results = self.search_similar(question, k=8)

        if not results:
            return {
                'answer': "Sorry, I couldn't find relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0
            }

        # Combine relevant passages
        context_parts = []
        sources = []
        total_score = 0

        for text, score, filename in results[:5]:
            context_parts.append(f"[From {filename}]: {text}")
            if filename not in sources:
                sources.append(filename)
            total_score += score

        context = "\n\n".join(context_parts)
        answer = self.generate_answer(question, context)

        return {
            'answer': answer,
            'sources': sources,
            'confidence': min(total_score / 5, 1.0),  # Fixed: always divide by 5
            'context': context[:1000] + "..." if len(context) > 1000 else context
        }

    def generate_answer(self, question: str, context: str) -> str:
        answer_parts = []

        if "grundzulage" in question.lower():
            amounts = re.findall(r'(\d+(?:,\d+)?)\s*(?:euro|€|\w*gulage)', context.lower())
            if amounts:
                # Remove duplicates while preserving order
                unique_amounts = list(dict.fromkeys(amounts))
                answer_parts.append(f"Based on the documents, relevant amounts mentioned: {', '.join(unique_amounts)}")

        elif "steuerlich" in question.lower() or "besteuert" in question.lower():
            tax_sentences = []
            for sentence in context.split('.'):
                if any(word in sentence.lower() for word in ['steuer', 'besteuert', 'steuerpflichtig', 'besteuerung']):
                    sentence_clean = sentence.strip()
                    # Avoid duplicate sentences
                    if sentence_clean and sentence_clean not in tax_sentences:
                        tax_sentences.append(sentence_clean)
            if tax_sentences:
                answer_parts.extend(tax_sentences[:3])

        elif "direktversicherung" in question.lower() or "pensionskasse" in question.lower():
            relevant_sentences = []
            for sentence in context.split('.'):
                if any(word in sentence.lower() for word in ['direktversicherung', 'pensionskasse', 'pensionsfonds', 'auszahlung']):
                    sentence_clean = sentence.strip()
                    # Avoid duplicate sentences
                    if sentence_clean and sentence_clean not in relevant_sentences:
                        relevant_sentences.append(sentence_clean)
            if relevant_sentences:
                answer_parts.extend(relevant_sentences[:3])

        if not answer_parts:
            sentences = context.split('.')
            unique_sentences = []
            for sent in sentences[:5]:  # Check more sentences initially
                sent_clean = sent.strip()
                if len(sent_clean) > 20 and sent_clean not in unique_sentences:
                    unique_sentences.append(sent_clean)
                    if len(unique_sentences) >= 3:  # Stop when we have 3 unique sentences
                        break
            answer_parts = unique_sentences

        if answer_parts:
            return "Based on the analyzed documents:\n\n" + "\n\n".join(answer_parts)
        else:
            return f"I found relevant passages but couldn't extract a specific answer. Please check the context provided."

# Flask App
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global variables
rag_pipeline = None
processing_status = {"status": "idle", "message": "Ready", "progress": 0}
pdf_directory = "./pdfs"

@app.route('/')
def index():
    """Main page"""
    # Check for PDF files
    pdf_files = []
    if os.path.exists(pdf_directory):
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]

    return render_template('index.html',
                         pdf_files=pdf_files,
                         pdf_count=len(pdf_files),
                         has_index=rag_pipeline is not None)

@app.route('/process', methods=['POST'])
def process_pdfs():
    """Process PDF files"""
    global rag_pipeline, processing_status

    if not os.path.exists(pdf_directory):
        return jsonify({'error': 'PDF directory does not exist'})

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return jsonify({'error': 'No PDF files found'})

    # Start processing in background thread
    def process_thread():
        global rag_pipeline, processing_status
        try:
            processing_status = {"status": "processing", "message": "Initializing...", "progress": 10}

            rag_pipeline = RAGPipeline()

            def progress_callback(message):
                processing_status["message"] = message
                processing_status["progress"] = min(processing_status["progress"] + 15, 90)

            pdf_count, chunk_count = rag_pipeline.process_pdfs(pdf_directory, progress_callback)

            processing_status = {
                "status": "complete",
                "message": f"Processed {pdf_count} PDFs, created {chunk_count} chunks",
                "progress": 100
            }

        except Exception as e:
            processing_status = {"status": "error", "message": str(e), "progress": 0}

    thread = threading.Thread(target=process_thread)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True})

@app.route('/status')
def get_status():
    """Get processing status"""
    return jsonify(processing_status)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer a question"""
    global rag_pipeline

    if not rag_pipeline:
        return jsonify({'error': 'Please process PDFs first'})

    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Please provide a question'})

    try:
        result = rag_pipeline.answer_question(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/quick-question/<int:q_id>')
def quick_question(q_id):
    """Handle quick questions"""
    questions = [
        "Wie hoch ist die Grundzulage?",
        "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
        "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
    ]

    if q_id < 1 or q_id > len(questions):
        return jsonify({'error': 'Invalid question ID'})

    if not rag_pipeline:
        return jsonify({'error': 'Please process PDFs first'})

    try:
        question = questions[q_id - 1]
        result = rag_pipeline.answer_question(question)
        result['question'] = question
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Create pdfs directory if it doesn't exist
    os.makedirs(pdf_directory, exist_ok=True)

    print("🚀 Starting RAG Pipeline Web Interface...")
    print("📁 Make sure to add your PDF files to the 'pdfs' directory")
    print("🌐 Open your browser and go to: http://localhost:8888")
    print("⚡ Press Ctrl+C to stop the server")

    app.run(debug=True, host='0.0.0.0', port=8888)
