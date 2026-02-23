import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Compatibility across LangChain versions
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

PDF_FOLDER = os.getenv("PDF_FOLDER", "medical_pdfs")
VECTOR_DB = os.getenv("VECTOR_DB", "vector_db")
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.jsonl")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)
 

def normalize_disease_name(filename: str) -> str:
    return filename.replace(".pdf", "").replace("_", " ").strip()


def stable_chunk_id(source_file: str, page, text: str) -> str:
    raw = f"{source_file}|{page}|{text}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def main():
    if not os.path.isdir(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    docs = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in: {PDF_FOLDER}")

    for file in pdf_files:
        file_path = os.path.join(PDF_FOLDER, file)
        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()

        disease_name = normalize_disease_name(file)

        for d in pdf_docs:
            d.metadata["disease"] = disease_name
            d.metadata["source_file"] = file
            if "page" not in d.metadata:
                d.metadata["page"] = None
            docs.append(d)

    # Token-aware splitter when available
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    chunks = splitter.split_documents(docs)

    for c in chunks:
        c.metadata["chunk_id"] = stable_chunk_id(
            c.metadata.get("source_file", "unknown.pdf"),
            c.metadata.get("page", None),
            c.page_content
        )

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    ensure_dir(VECTOR_DB)
    vectorstore.save_local(VECTOR_DB)

    ensure_dir(os.path.dirname(CORPUS_PATH))
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "text": c.page_content,
                "metadata": c.metadata
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(" Ingestion completed!")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"FAISS saved to: {VECTOR_DB}")
    print(f"BM25 corpus saved to: {CORPUS_PATH}")
    
if __name__ == "__main__":
    main()