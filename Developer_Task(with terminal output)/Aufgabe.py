import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

class RAGPipeline:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize RAG Pipeline with sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        self.chunks = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF
        """
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
        """
        Split text into overlapping chunks for better context preservation
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())

        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk_size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]  # Filter very short chunks

    def process_pdfs(self, pdf_directory: str):
        """
        Process all PDF files in directory and create vector database
        """
        print("Processing PDF files...")

        # Extract text from all PDFs
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                print(f"Processing: {filename}")

                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    chunks = self.chunk_text(text)
                    for chunk in chunks:
                        self.documents.append({
                            'filename': filename,
                            'text': chunk
                        })
                        self.chunks.append(chunk)

        print(f"Created {len(self.chunks)} text chunks from {len(set([doc['filename'] for doc in self.documents]))} PDF files")

        # Create embeddings
        print("Creating embeddings...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        print("Vector database created successfully!")

    def search_similar(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for similar documents using vector similarity
        """
        if self.index is None:
            raise ValueError("Index not created. Please process PDFs first.")

        # Create query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):  # Ensure valid index
                results.append((
                    self.documents[idx]['text'],
                    float(score),
                    self.documents[idx]['filename']
                ))

        return results

    def answer_question(self, question: str, max_context_length: int = 2000) -> str:
        """
        Answer question using retrieved relevant passages
        """
        print(f"\nQuestion: {question}")
        print("Searching for relevant passages...")

        # Retrieve relevant passages
        results = self.search_similar(question, k=8)

        if not results:
            return "Sorry, I couldn't find relevant information to answer your question."

        # Combine relevant passages
        context_parts = []
        total_length = 0

        for text, score, filename in results:
            if total_length + len(text) <= max_context_length:
                context_parts.append(f"[From {filename}]: {text}")
                total_length += len(text)
                print(f"Found relevant passage (score: {score:.3f}) from {filename}")
            else:
                break

        if not context_parts:
            return "No relevant passages found."

        context = "\n\n".join(context_parts)

        # Simple answer generation based on context
        answer = self.generate_answer(question, context)

        return answer

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer from context (simplified version)
        In a production system, you would use a language model here
        """
        # This is a simplified approach - in practice you'd use GPT/LLM here
        answer_parts = []

        # Look for specific patterns in the context based on the question type
        if "grundzulage" in question.lower():
            # Look for amounts/numbers that might be the basic allowance
            import re
            amounts = re.findall(r'(\d+(?:,\d+)?)\s*(?:euro|€|\w*gulage)', context.lower())
            if amounts:
                answer_parts.append(f"Based on the documents, relevant amounts mentioned: {', '.join(amounts)}")

        elif "steuerlich" in question.lower() or "besteuert" in question.lower():
            # Look for tax-related information
            tax_sentences = []
            for sentence in context.split('.'):
                if any(word in sentence.lower() for word in ['steuer', 'besteuert', 'steuerpflichtig', 'besteuerung']):
                    tax_sentences.append(sentence.strip())
            if tax_sentences:
                answer_parts.extend(tax_sentences[:3])  # Top 3 relevant sentences

        elif "direktversicherung" in question.lower() or "pensionskasse" in question.lower():
            # Look for information about direct insurance and pension funds
            relevant_sentences = []
            for sentence in context.split('.'):
                if any(word in sentence.lower() for word in ['direktversicherung', 'pensionskasse', 'pensionsfonds', 'auszahlung']):
                    relevant_sentences.append(sentence.strip())
            if relevant_sentences:
                answer_parts.extend(relevant_sentences[:3])

        if not answer_parts:
            # Fallback: return most relevant parts of context
            sentences = context.split('.')
            answer_parts = [sent.strip() for sent in sentences[:3] if len(sent.strip()) > 20]

        if answer_parts:
            return "Based on the analyzed documents:\n\n" + "\n\n".join(answer_parts)
        else:
            return f"I found relevant passages but couldn't extract a specific answer. Here's what I found:\n\n{context[:500]}..."

    def save_index(self, filepath: str):
        """Save the vector index and documents"""
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")

            # Save documents and metadata
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'chunks': self.chunks,
                    'dimension': self.dimension
                }, f)
            print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        """Load the vector index and documents"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")

            # Load documents and metadata
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.chunks = data['chunks']
                self.dimension = data['dimension']
            print(f"Index loaded from {filepath}")
        except Exception as e:
            print(f"Error loading index: {e}")

def main():
    """
    Main function to demonstrate the RAG pipeline
    """
    # Initialize RAG pipeline
    rag = RAGPipeline()

    # Directory containing PDF files
    pdf_directory = "pdfs"  # Change this to your PDF directory

    # Check if directory exists
    if not os.path.exists(pdf_directory):
        print(f"Creating directory: {pdf_directory}")
        os.makedirs(pdf_directory)
        print(f"Please add your PDF files to the '{pdf_directory}' directory and run the script again.")
        return

    # Check if there are PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_directory}' directory.")
        print("Please add PDF files and run the script again.")
        return

    # Process PDFs and create vector database
    try:
        rag.process_pdfs(pdf_directory)

        # Save the index for future use
        rag.save_index("pension_documents_index")

        # Test questions from the assignment
        questions = [
            "Wie hoch ist die Grundzulage?",
            "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
            "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
        ]

        print("\n" + "="*70)
        print("ANSWERING QUESTIONS")
        print("="*70)

        for question in questions:
            print("\n" + "-"*70)
            answer = rag.answer_question(question)
            print(f"\nAnswer: {answer}")
            print("-"*70)

        # Interactive mode
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("You can now ask questions about the documents.")
        print("Type 'quit' to exit.")

        while True:
            user_question = input("\nYour question: ").strip()
            if user_question.lower() in ['quit', 'exit', 'q']:
                break
            if user_question:
                answer = rag.answer_question(user_question)
                print(f"\nAnswer: {answer}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed the required packages:")
        print("pip install PyMuPDF sentence-transformers faiss-cpu numpy")

if __name__ == "__main__":
    main()
