import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma

# from langchain_nomic.embeddings import NomicEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

CHROMA_DB_DIR = "./chroma-db"
CHROMA_COLLECTION_NAME = "grumatic_heka"
# CHROMA_EMBEDDING = NomicEmbeddings(
#     model="nomic-embed-text-v1.5", inference_mode="local"
# )
CHROMA_EMBEDDING = OpenAIEmbeddings(model="text-embedding-3-large")

urls = [
    "https://www.heka.so/",
    "https://www.heka.so/feature/billing",
    "https://www.heka.so/feature/customer",
    "https://www.heka.so/feature/billing-portal",
    "https://www.heka.so/about",
    "https://grumatic.notion.site/heka-Guide-document-f9920ef200184661a32e5239553e751f",
]

pdfs = [
    "./files/grumatic_intro_202409_for_comtec.pdf",
    "./files/heka_v2_documentation.pdf",
    "./files/HK-heka_MSP_-_Product_Requirement_Document-250924-054956.pdf",
    "./files/HK-Cloud_Service_Invoicing_Workflow-250924-055111.pdf",
    "./files/HK-Functional_Requirements_for_Customers-250924-055210.pdf",
    "./files/HK-Functional_Requirements_for_MSP-250924-055138.pdf",
    "./files/HK-Invoice_Requirements-250924-055245.pdf",
    "./files/finops_edu_and_intro_v1_1_for_harim.pdf",
]
# [
#     "./files/dev_rules.pdf",
#     "./files/grumatic_intro_202406.pdf",
# ]

DO_INJEST = False

if DO_INJEST:
    web_docs = [WebBaseLoader(url).load() for url in urls]
    web_docs_list = [item for sublist in web_docs for item in sublist]
    web_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=30, encoding_name="cl100k_base"
    )
    web_doc_splits = web_text_splitter.split_documents(web_docs_list)

    pdf_docs = [PyPDFLoader(pdf).load() for pdf in pdfs]
    pdf_docs_list = [item for sublist in pdf_docs for item in sublist]
    pdf_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600, chunk_overlap=60, encoding_name="cl100k_base"
    )
    pdf_doc_splits = pdf_text_splitter.split_documents(pdf_docs_list)

    # import pdb;pdb.set_trace()
    doc_splits = web_doc_splits + pdf_doc_splits

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding=CHROMA_EMBEDDING,
    )
else:
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=CHROMA_EMBEDDING,
    )

retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI()

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)

sample_questions = [
    "what is heka?",
    "what kinds of features heka provides?",
    "what is the vision of Grumatic?",
    "what is the FinOps Strategy of Grumatic?"
]

if __name__ == "__main__":
    for q in sample_questions:
        print("-----------------\n")
        print(f"User: {q}\n")
        res = chain.invoke(q)
        print(f"AI: {res}\n")
