from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

def load_documents():
    dir_loader =DirectoryLoader(
        "../data/pdfs",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False
    )
    return dir_loader.load()