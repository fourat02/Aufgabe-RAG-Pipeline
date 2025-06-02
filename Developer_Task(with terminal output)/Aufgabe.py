import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
from pathlib import Path
import json

warnings.filterwarnings("ignore")

@dataclass
class RAGConfig:
    """Configuration class for RAG Pipeline"""
    model_name: str = 'all-MiniLM-L6-v2'
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_context_length: int = 2000
    similarity_threshold: float = 0.3
    max_results: int = 8
    min_chunk_length: int = 20

class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG Pipeline with configurable parameters
        """
        self.config = config or RAGConfig()
        self.logger = self._setup_logger()

        try:
            self.model = SentenceTransformer(self.config.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Loaded model: {self.config.model_name}, dimension: {self.dimension}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise

        self.index = None
        self.document_metadata = []  # Only store metadata, not full text
        self.chunks = []

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('RAGPipeline')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF with better error handling and validation
        """
        try:
            if not os.path.exists(pdf_path):
                self.logger.error(f"PDF file not found: {pdf_path}")
                return None

            doc = fitz.open(pdf_path)
            text = ""
            page_count = 0

            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += page_text + "\n"
                        page_count += 1
                except Exception as e:
                    self.logger.warning(f"Error extracting text from page {page_num} in {pdf_path}: {e}")
                    continue

            doc.close()

            if not text.strip():
                self.logger.warning(f"No text extracted from {pdf_path}")
                return None

            self.logger.info(f"Extracted text from {page_count} pages in {os.path.basename(pdf_path)}")
            return text

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks with proper word-based overlap
        """
        if not text or not text.strip():
            return []

        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?;:()\-€$%]', ' ', text)  # Keep basic punctuation

        # Split into sentences for better chunk boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])'
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence exceeds chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())

                # Create overlap for next chunk (word-based, not character-based)
                words = current_chunk.split()
                overlap_words = min(self.config.chunk_overlap, len(words))
                overlap_text = ' '.join(words[-overlap_words:]) if overlap_words > 0 else ""

                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                current_chunk = potential_chunk

        # Add the final chunk
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Filter out chunks that are too short
        valid_chunks = [chunk for chunk in chunks if len(chunk) >= self.config.min_chunk_length]

        self.logger.info(f"Created {len(valid_chunks)} chunks from text")
        return valid_chunks

    def process_pdfs(self, pdf_directory: str) -> bool:
        """
        Process all PDF files in directory with comprehensive error handling
        """
        pdf_path = Path(pdf_directory)

        if not pdf_path.exists():
            self.logger.error(f"Directory does not exist: {pdf_directory}")
            return False

        if not pdf_path.is_dir():
            self.logger.error(f"Path is not a directory: {pdf_directory}")
            return False

        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            self.logger.error(f"No PDF files found in {pdf_directory}")
            return False

        self.logger.info(f"Found {len(pdf_files)} PDF files to process")

        processed_files = 0
        total_chunks = 0

        # Reset storage
        self.document_metadata = []
        self.chunks = []

        for pdf_file in pdf_files:
            self.logger.info(f"Processing: {pdf_file.name}")

            text = self.extract_text_from_pdf(str(pdf_file))
            if not text:
                self.logger.warning(f"Skipping {pdf_file.name} - no text extracted")
                continue

            chunks = self.chunk_text(text)
            if not chunks:
                self.logger.warning(f"Skipping {pdf_file.name} - no valid chunks created")
                continue

            # Store metadata and chunks
            for i, chunk in enumerate(chunks):
                self.document_metadata.append({
                    'filename': pdf_file.name,
                    'chunk_id': i,
                    'chunk_length': len(chunk)
                })
                self.chunks.append(chunk)

            processed_files += 1
            total_chunks += len(chunks)
            self.logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")

        if processed_files == 0:
            self.logger.error("No PDF files were successfully processed")
            return False

        self.logger.info(f"Successfully processed {processed_files} files, created {total_chunks} chunks")

        # Create embeddings and index
        return self._create_vector_index()

    def _create_vector_index(self) -> bool:
        """
        Create FAISS vector index from chunks
        """
        if not self.chunks:
            self.logger.error("No chunks available for indexing")
            return False

        try:
            self.logger.info("Creating embeddings...")
            embeddings = self.model.encode(
                self.chunks,
                show_progress_bar=True,
                batch_size=32  # Process in batches to avoid memory issues
            )

            if embeddings is None or len(embeddings) == 0:
                self.logger.error("Failed to create embeddings")
                return False

            # Create FAISS index with cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)

            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.astype('float32')
            faiss.normalize_L2(embeddings_normalized)

            # Add to index
            self.index.add(embeddings_normalized)

            self.logger.info(f"Vector index created with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            self.logger.error(f"Error creating vector index: {e}")
            return False

    def search_similar(self, query: str, k: Optional[int] = None) -> List[Tuple[str, float, str]]:
        """
        Search for similar documents with similarity threshold filtering
        """
        if self.index is None:
            raise ValueError("Index not created. Please process PDFs first.")

        if not query or not query.strip():
            return []

        k = k or self.config.max_results

        try:
            # Create and normalize query embedding
            query_embedding = self.model.encode([query.strip()])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)

            # Ensure k doesn't exceed available vectors
            max_k = min(k, self.index.ntotal)
            if max_k == 0:
                self.logger.warning("No vectors in index")
                return []

            # Search
            scores, indices = self.index.search(query_embedding, max_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Skip invalid indices
                if idx < 0:
                    continue

                # Filter by similarity threshold
                if score < self.config.similarity_threshold:
                    continue

                # Ensure idx is within bounds for both arrays
                if idx < len(self.chunks) and idx < len(self.document_metadata):
                    results.append((
                        self.chunks[idx],
                        float(score),
                        self.document_metadata[idx]['filename']
                    ))
                else:
                    self.logger.warning(f"Index {idx} out of bounds (chunks: {len(self.chunks)}, metadata: {len(self.document_metadata)})")

            self.logger.info(f"Found {len(results)} relevant passages above threshold {self.config.similarity_threshold}")
            return results

        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def answer_question(self, question: str) -> str:
        """
        Answer question using retrieved relevant passages with improved logic
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        self.logger.info(f"Processing question: {question}")

        # Retrieve relevant passages
        results = self.search_similar(question)

        if not results:
            return "I couldn't find relevant information in the documents to answer your question. Please try rephrasing or asking about a different topic."

        # Build context from results
        context_parts = []
        total_length = 0
        used_files = set()

        for text, score, filename in results:
            # Prioritize diversity of sources
            context_length = len(text)

            if total_length + context_length <= self.config.max_context_length:
                context_parts.append({
                    'text': text,
                    'filename': filename,
                    'score': score
                })
                total_length += context_length
                used_files.add(filename)
                self.logger.info(f"Using passage (score: {score:.3f}) from {filename}")
            else:
                break

        if not context_parts:
            return "Found relevant passages but they were too long to process effectively."

        # Generate answer
        answer = self._generate_contextual_answer(question, context_parts)

        # Add source information
        sources = sorted(used_files)
        source_info = f"\n\nSources: {', '.join(sources)}"

        return answer + source_info

    def _generate_contextual_answer(self, question: str, context_parts: List[Dict]) -> str:
        """
        Generate answer from context using improved heuristics
        """
        question_lower = question.lower()

        # Combine all context
        full_context = "\n\n".join([part['text'] for part in context_parts])

        # Look for specific answer patterns based on question type
        answer_sentences = []

        # German-specific question patterns
        if any(word in question_lower for word in ['wie hoch', 'höhe', 'betrag', 'summe']):
            # Look for amounts/numbers
            amount_pattern = r'(\d+(?:[.,]\d+)?)\s*(?:euro|€|prozent|%)'
            amounts = re.findall(amount_pattern, full_context, re.IGNORECASE)
            if amounts:
                answer_sentences.append(f"Relevante Beträge aus den Dokumenten: {', '.join(set(amounts))}")

        if any(word in question_lower for word in ['steuerlich', 'besteuert', 'besteuerung', 'steuer']):
            # Tax-related information
            for part in context_parts:
                sentences = re.split(r'[.!?]+', part['text'])
                for sentence in sentences:
                    if any(tax_word in sentence.lower() for tax_word in
                          ['steuer', 'besteuert', 'steuerpflichtig', 'besteuerung', 'steuerfrei']):
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 10 and clean_sentence not in answer_sentences:
                            answer_sentences.append(clean_sentence)
                            if len(answer_sentences) >= 3:
                                break
                if len(answer_sentences) >= 3:
                    break

        if any(word in question_lower for word in ['direktversicherung', 'pensionskasse', 'pensionsfonds']):
            # Insurance/pension specific information
            for part in context_parts:
                sentences = re.split(r'[.!?]+', part['text'])
                for sentence in sentences:
                    if any(ins_word in sentence.lower() for ins_word in
                          ['direktversicherung', 'pensionskasse', 'pensionsfonds', 'auszahlung', 'leistung']):
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 10 and clean_sentence not in answer_sentences:
                            answer_sentences.append(clean_sentence)
                            if len(answer_sentences) >= 3:
                                break
                if len(answer_sentences) >= 3:
                    break

        # If no specific patterns found, extract most relevant sentences
        if not answer_sentences:
            all_sentences = []
            for part in context_parts:
                sentences = re.split(r'[.!?]+', part['text'])
                for sentence in sentences:
                    clean_sentence = sentence.strip()
                    if len(clean_sentence) > 20:  # Minimum meaningful length
                        all_sentences.append((clean_sentence, part['score']))

            # Sort by relevance score and take top sentences
            all_sentences.sort(key=lambda x: x[1], reverse=True)
            answer_sentences = [sent[0] for sent in all_sentences[:3]]

        if answer_sentences:
            return "Basierend auf den analysierten Dokumenten:\n\n" + "\n\n".join(answer_sentences)
        else:
            # Final fallback
            best_context = context_parts[0]['text'][:500]
            return f"Ich habe relevante Informationen gefunden, kann aber keine spezifische Antwort extrahieren. Hier sind die relevantesten Passagen:\n\n{best_context}..."

    def save_index(self, filepath: str) -> bool:
        """Save the vector index and metadata with validation"""
        try:
            if self.index is None:
                self.logger.error("No index to save")
                return False

            # Ensure directory exists
            save_path = Path(filepath).parent
            save_path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")

            # Save metadata and configuration
            metadata = {
                'document_metadata': self.document_metadata,
                'chunks': self.chunks,
                'dimension': self.dimension,
                'config': {
                    'model_name': self.config.model_name,
                    'chunk_size': self.config.chunk_size,
                    'chunk_overlap': self.config.chunk_overlap,
                    'max_context_length': self.config.max_context_length,
                    'similarity_threshold': self.config.similarity_threshold,
                    'max_results': self.config.max_results,
                    'min_chunk_length': self.config.min_chunk_length
                }
            }

            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)

            # Also save as JSON for readability
            with open(f"{filepath}_info.json", 'w', encoding='utf-8') as f:
                info = {
                    'total_chunks': len(self.chunks),
                    'total_documents': len(set(meta['filename'] for meta in self.document_metadata)),
                    'config': metadata['config']
                }
                json.dump(info, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Index and metadata saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
            return False

    def load_index(self, filepath: str) -> bool:
        """Load the vector index and metadata with validation"""
        try:
            # Check if files exist
            if not os.path.exists(f"{filepath}.faiss"):
                self.logger.error(f"Index file not found: {filepath}.faiss")
                return False

            if not os.path.exists(f"{filepath}_metadata.pkl"):
                self.logger.error(f"Metadata file not found: {filepath}_metadata.pkl")
                return False

            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")

            # Load metadata
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)

                self.document_metadata = metadata.get('document_metadata', [])
                self.chunks = metadata.get('chunks', [])
                self.dimension = metadata.get('dimension', self.dimension)

                # Update config if saved
                if 'config' in metadata:
                    saved_config = metadata['config']
                    for key, value in saved_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)

            # Validate loaded data
            if len(self.document_metadata) != len(self.chunks):
                self.logger.warning(f"Metadata and chunks count mismatch: metadata={len(self.document_metadata)}, chunks={len(self.chunks)}")
                # Try to fix the mismatch
                min_len = min(len(self.document_metadata), len(self.chunks))
                if min_len > 0:
                    self.document_metadata = self.document_metadata[:min_len]
                    self.chunks = self.chunks[:min_len]
                    self.logger.info(f"Truncated to {min_len} items to fix mismatch")
                else:
                    self.logger.error("Cannot fix mismatch - no valid data")
                    return False

            if self.index.ntotal != len(self.chunks):
                self.logger.warning(f"Index size and chunks count mismatch: index={self.index.ntotal}, chunks={len(self.chunks)}")
                # If index has more vectors than chunks, we need to rebuild
                if self.index.ntotal > len(self.chunks):
                    self.logger.warning("Index has more vectors than chunks - this may cause search errors")
                    # Truncate search results to available chunks
                    self.config.max_results = min(self.config.max_results, len(self.chunks))

            self.logger.info(f"Successfully loaded index with {self.index.ntotal} vectors from {filepath}")
            self.logger.info(f"Available chunks: {len(self.chunks)}, metadata entries: {len(self.document_metadata)}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading index: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

def main():
    """
    Main function with improved error handling and user experience
    """
    # Initialize with custom configuration
    config = RAGConfig(
        chunk_size=600,
        chunk_overlap=75,
        max_context_length=2500,
        similarity_threshold=0.25,
        max_results=10
    )

    rag = RAGPipeline(config)

    # Directory setup
    pdf_directory = "pdfs"
    index_name = "pension_documents_index"

    # Check if we can load existing index
    if os.path.exists(f"{index_name}.faiss"):
        print("Found existing index. Loading...")
        if rag.load_index(index_name):
            print("Index loaded successfully!")
        else:
            print("Failed to load existing index. Will create new one.")
            rag.index = None

    # Process PDFs if no index loaded
    if rag.index is None:
        print(f"Processing PDFs from '{pdf_directory}' directory...")

        if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory)
            print(f"Created '{pdf_directory}' directory. Please add your PDF files and run again.")
            return

        if not rag.process_pdfs(pdf_directory):
            print("Failed to process PDFs. Please check the logs and try again.")
            return

        # Save the index
        if rag.save_index(index_name):
            print("Index saved successfully!")
        else:
            print("Warning: Failed to save index.")

    # Test questions
    test_questions = [
        "Wie hoch ist die Grundzulage?",
        "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
        "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?"
    ]

    print("\n" + "="*80)
    print("ANSWERING TEST QUESTIONS")
    print("="*80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        print("\nA:", rag.answer_question(question))
        print("-" * 80)

    # Interactive mode
    print(f"\n{'='*80}")
    print("INTERACTIVE MODE")
    print("="*80)
    print("Ask questions about the documents. Type 'quit', 'exit', or 'q' to exit.")
    print("Type 'help' for available commands.")

    while True:
        try:
            user_input = input("\nYour question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                print("Available commands:")
                print("- Ask any question about the documents")
                print("- 'stats' - Show system statistics")
                print("- 'config' - Show current configuration")
                print("- 'quit' or 'exit' - Exit the program")
                continue

            if user_input.lower() == 'stats':
                print(f"Total documents: {len(set(meta['filename'] for meta in rag.document_metadata))}")
                print(f"Total chunks: {len(rag.chunks)}")
                print(f"Total metadata entries: {len(rag.document_metadata)}")
                print(f"Index size: {rag.index.ntotal if rag.index else 0}")
                print(f"Model dimension: {rag.dimension}")
                print(f"Similarity threshold: {rag.config.similarity_threshold}")
                continue

            if user_input.lower() == 'config':
                print(f"Model: {rag.config.model_name}")
                print(f"Chunk size: {rag.config.chunk_size}")
                print(f"Similarity threshold: {rag.config.similarity_threshold}")
                print(f"Max results: {rag.config.max_results}")
                continue

            # Answer the question
            answer = rag.answer_question(user_input)
            print(f"\nAnswer: {answer}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing question: {e}")
            continue

if __name__ == "__main__":
    main()
