import os
import math
import numexpr
import re
from langchain_core.tools import tool, BaseTool
from langchain_community.tools import DuckDuckGoSearchResults, ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from agent import heka_database
from typing import List

from langchain_chroma import Chroma

# from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

web_search = DuckDuckGoSearchResults(name="WebSearch")

# Kinda busted since it doesn't return links
arxiv_search = ArxivQueryRun(name="ArxivSearch")


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"

# for heka
heka_user_db = heka_database.get_database_collection(heka_database.collections["user"])
heka_company_db = heka_database.get_database_collection(
    heka_database.collections["company"]
)
heka_organization_db = heka_database.get_database_collection(
    heka_database.collections["organization"]
)
heka_aws_invoice_db = heka_database.get_database_collection(
    heka_database.collections["aws_invoice"]
)


@tool
def get_all_companies(domain: str = "heka") -> List[dict]:
    """Get all companies in heka"""
    try:
        cursor = heka_company_db.find({"delete_at": None})
        result = []
        for company in cursor:
            result.append({"name": company["name"], "id": company["uid"]})
        return result
    except:
        return {"error": "Invalid query"}


@tool
def query_organizations_by_company_id(
    company_id: str = "grumatic-default",
) -> List[dict]:
    """Query about heka organization information by company id in heka"""
    try:
        cursor = heka_organization_db.find(
            {"delete_at": None, "company_id": company_id}
        )
        result = []
        for org in cursor:
            filtered_org = {
                key: org[key]
                for key in [
                    "company_id",
                    "name",
                    "representative",
                    "business_location",
                    "telephone",
                    # "cloud_accounts",
                    # "billing_custom",
                    "last_invoiced",
                ]
            } | {"id": org["uid"]}
            result.append(filtered_org)
        return result
    except:
        return {"error": "Invalid query"}


@tool
def query_user_by_email(email: str) -> dict:
    """Query about user information by email in heka"""
    try:
        result = heka_user_db.find_one({"email": email})
        return result
    except:
        return {"error": "Invalid query"}


@tool
def query_aws_invoice_by_invoice_id(invoice_id: str) -> dict:
    """Query about AWS invoice information by invoice id in heka"""
    try:
        result = heka_aws_invoice_db.find_one({"invoice_id": invoice_id})
        return result.pop("data")
    except:
        return {"error": "Invalid query"}


@tool
def query_aws_invoice_by_organization_id(
    organization_id: str, status: str = "Origin"
) -> List[dict]:
    """Query about a list of AWS invoice by organization id and status in heka"""
    try:
        cursor = heka_aws_invoice_db.find(
            {
                "delete_at": None,
                "status": status,
                "organization_id": organization_id,
            }
        )
        result = []
        for item in cursor:
            filtered_item = {
                key: item[key]
                for key in [
                    "company_id",
                    "organization_id",
                    "account_id",
                    "total_cost",
                    "org_total_cost",
                ]
            } | {"id": item["invoice_id"]}
            result.append(filtered_item)
        return result
    except:
        return {"error": "Invalid query"}

    # test
    # query_user({'email': "grmt-root@grumatic.com"})


if os.getenv("LANGSMITH_LANGGRAPH_DESKTOP"):
    # LangGraph Studio Docker Container
    CHROMA_DB_DIR = "/deps/__outer_agent/agent/chroma-db"
else:
    # Agent-Service-Tool Docker Container
    CHROMA_DB_DIR = "/app/agent/chroma-db"

CHROMA_COLLECTION_NAME = "rag-chroma"

# from langchain_nomic.embeddings import NomicEmbeddings
# CHROMA_EMBEDDING = NomicEmbeddings(
#     model="nomic-embed-text-v1.5", inference_mode="local"
# )

CHROMA_EMBEDDING = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    collection_name=CHROMA_COLLECTION_NAME,
    persist_directory=CHROMA_DB_DIR,
    embedding_function=CHROMA_EMBEDDING,
)

retriever = vectorstore.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "grumatic_heka_retriever",
    "Searches and returns excerpts from heka solutions by Grumatic (그루매틱 헤카 솔루션)",
)

tools = [
    retriever_tool,
    calculator,
    #    arxiv_search,
    #    web_search,
    TavilySearchResults(max_results=1),
    get_all_companies,
    query_organizations_by_company_id,
    query_user_by_email,
    query_aws_invoice_by_invoice_id,
    query_aws_invoice_by_organization_id,
]
