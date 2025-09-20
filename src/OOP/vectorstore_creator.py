from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

import torch
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

console = Console(force_terminal=True)


class VectorStoreCreator(BaseModel):
    """
        A class to create vectorstore.
        :param rag_path: Path to the RAG folder. If provided RAG becomes persisted and database is saved in provided folder.
        :type rag_path:
        :param verbose: Verbose mode. Default is False.
        :type verbose: bool
    """
    model_config = ConfigDict(use_attribute_docstrings=True)
    rag_path: str = Field(strict=True, frozen=True, description="Path to the rag file.")
    verbose: bool = Field(strict=True, frozen=True, default=False, description="Verbose mode. Default is False.")

    def __init__(self, rag_path: str, verbose: bool):
        super().__init__(
            rag_path=rag_path,
            verbose=verbose
        )

    def create_vectorstore(self, chunked_story: list[Document]) -> Chroma:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        if self.rag_path == "":
            vectorstore = Chroma().from_documents(chunked_story, embeddings)
        else:
            vectorstore = Chroma(persist_directory=self.rag_path).from_documents(chunked_story, embeddings)
        if self.verbose:
            console.log(f"Vectorstore created!", style="green")
        return vectorstore
