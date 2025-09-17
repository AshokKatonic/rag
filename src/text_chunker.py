from langchain.text_splitter import CharacterTextSplitter
from . import settings

def create_text_splitter(
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP
):
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

def split_into_chunks(text, text_splitter=None):
    if text_splitter is None:
        text_splitter = create_text_splitter()
    return text_splitter.split_text(text)