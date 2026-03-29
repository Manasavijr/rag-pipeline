import logging
from pathlib import Path
from typing import List, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load(self, file_path: str) -> Tuple[List[Document], int]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in settings.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info(f"Loading document: {path.name} ({suffix})")

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            pages = len(docs)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
            pages = 1
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
            pages = len(docs)
        else:
            raise ValueError(f"Unsupported: {suffix}")

        for doc in docs:
            doc.metadata.update({
                "source": path.name,
                "file_path": str(path),
                "file_type": suffix,
            })

        chunks = self.splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {path.name}")
        return chunks, pages

    def load_text(self, text: str, source: str = "direct_input") -> List[Document]:
        doc = Document(page_content=text, metadata={"source": source, "file_type": "text"})
        return self.splitter.split_documents([doc])
