import os
import math
import numexpr
import re
import json
from langchain_core.tools import tool, BaseTool, create_retriever_tool
from langchain_core.runnables.config import RunnableConfig
from langchain.tools import StructuredTool
from langchain_community.tools import DuckDuckGoSearchResults, ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import List, Annotated, Union, Literal
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import InjectedState
from agent import heka_database
from agent.util import to_json_serializable_doc

#web_search = DuckDuckGoSearchResults(name="WebSearch")

# Kinda busted since it doesn't return links
arxiv_search = ArxivQueryRun(name="ArxivSearch")

# for chroma vectorstore retriever
if os.getenv("LANGSMITH_LANGGRAPH_DESKTOP"):
    # LangGraph Studio Docker Container
    CHROMA_DB_DIR = "/deps/__outer_agent/agent/chroma-db"
else:
    # Agent-Service-Tool Docker Container
    CHROMA_DB_DIR = "/app/agent/chroma-db"

# For local development, TODO: delete this
if os.getenv("HOME") == "/Users/xroger88":
    CHROMA_DB_DIR = "/Users/xroger88/Projects/Grumatic/llm/agent-service-toolkit/agent/chroma-db"

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
retriever_tool = create_retriever_tool(
    retriever,
    "grumatic_heka_retriever",
    "Searches and returns excerpts from heka solutions by Grumatic (그루매틱 헤카 솔루션)",
)

search_tools = [
    retriever_tool,
    arxiv_search,
    # web_search,
    TavilySearchResults(max_results=1),
]


# math tools
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

math_tools = [calculator]

# for heka database tools

@tool
def get_schema_info() -> dict:
    "Get schema information for heka NOSQL mongo database"
    schema = heka_database.get_schema()
    keys_to_extract = ["user", "company", "organization", "aws_invoice", "nhn_invoice"]
    # extracting keys using a for loop and conditional statement
    extracted_dict = {}
    for key, value in schema.items():
        if key in keys_to_extract:
            extracted_dict[key] = value

    return extracted_dict


@tool
def get_user_info(id: str = None, limit: int = 1) -> List[dict]:
    "Get user information in heka database"
    try:
        users = heka_database.get_user_info(id, limit)
        result = []
        for user in users:
            db = heka_database.get_database_collection("company")
            company_id = user["company_id"]
            company_info = db.find_one({"uid": company_id}, {"name": 1})
            company = f'{company_info["name"]}(id:{company_id})'
            org_list = []
            db = heka_database.get_database_collection("organization")
            for org_id in user["organizations"]:
                org_info = db.find_one({"uid": org_id},
                                       {"name": 1})
                org_list.append(f'{org_info["name"]}(id:{org_id})')
            additionals = {"full_name": f'{user["first_name"]} {user["last_name"]}',
                           "company": company,
                           "organizations": ",".join(org_list),
                           }
            result.append(user | additionals)
        return to_json_serializable_doc(result)
    except Exception as e:
        return {"error": f"Invalid query: {e}"}

@tool
def get_msp_info(id: str = None, limit: int = 1) -> List[dict]:
    "Get Managed Service Provider(MSP) information in heka database"
    try:
        companies = heka_database.get_company_info(id, limit)
        result = []
        for company in companies:
            db = heka_database.get_database_collection("organization")
            org_list = []
            cursor2 = db.find({"company_id": company["uid"]},
                              {"name": 1, "_id": 0})
            for org in cursor2:
                org_list.append(org["name"])
            additionals = {"customer_list": ",".join(org_list),
                           "customer_count": len(org_list)}
            result.append(company | additionals)
        return to_json_serializable_doc(result)
    except Exception as e:
        return {"error": f"Invalid query: {e}"}

@tool
def get_customer_info(id: str = None, limit: int = 1) -> List[dict]:
    "Get customer information in heka database"
    try:
        orgs = heka_database.get_organization_info(id, limit)
        result = []
        for org in orgs:
            invoice_collections = [cloud_account["csp"]+"_invoice" for cloud_account in org["cloud_accounts"]]
            for coll in invoice_collections:
                db = heka_database.get_database_collection(coll)
                cursor2 = db.find({"organization_id": org["uid"],
                                   "status": "Origin"},
                                  {"invoice_id": 1, "_id": 0},
                                  limit=0).sort({"invoice_id": -1}).limit(12)
                invoices = [item["invoice_id"] for item in cursor2]
                org = org | {f"{coll}s": invoices}
            result.append(org)

        # fallback that id is for msp company
        return to_json_serializable_doc(result) if result else get_msp_info.invoke({"id": id})
    except Exception as e:
        return {"error": f"Invalid query: {e}"}

@tool
def get_invoice_info(id: str, status: Literal["Origin", "Paid", "Unissued", "Invoiced"] = "Origin") -> List[dict]:
    "Get billing invoice information in heka database"
    try:
        result = []
        for coll in ["aws_invoice", "nhn_invoice"]:
            db = heka_database.get_database_collection(coll)
            for invoice in db.find({"$or": [{"invoice_id": id}, {"invoice_id": {"$regex": id, "$options": "i"}}],
                                    "status": status}, {"_id": 0}).sort({"invoice_id": -1}):
                result.append(invoice)
        return to_json_serializable_doc(result)
    except Exception as e:
        return {"error": f"Invalid query: {e}"}

@tool
def mongo_find(collection_name:str, query:dict = {}, projection:dict = {}, limit:int = 30) -> Union[List[dict], str]:
    """Useful to query some information using mongo find operation for NOSQL database.
    Make query and projection parameters according to mongo find specification.
    Use a limit parameter to restict the number of query results.
    """
    result = heka_database.find(collection_name, query=query, projection=projection, limit=limit)
    return to_json_serializable_doc(result)

@tool
def mongo_aggregate(collection_name:str, pipeline:List[dict] = []) -> Union[List[dict], str]:
    """Useful to query some information using mongo aggregate operation from NOSQL database.
    Make a pipeline parameter according to mongo aggregation specification.
    """
    result = heka_database.aggregate(collection_name, pipeline=pipeline)
    return to_json_serializable_doc(result)

@tool
def mongo_count_documents(collection_name:str, filter:dict = {}) -> int:
    """Useful to count the total number of documents in collection from NOSQL database.
    Make a filter parameter to count documents according to mongo specification.
    """
    result = heka_database.count_documents(collection_name)
    return to_json_serializable_doc(result)

mongo_tools = [mongo_find, mongo_aggregate, mongo_count_documents]

@tool
def mongo_query(user_question:str) -> List[dict] | dict | str:
    #, state: Annotated[dict, InjectedState], config: RunnableConfig):
    "Perform mongo database query for user question by using operations like 'find', 'aggregate', 'count_documents'. Used for complex NOSQL query operations. Provide the user question which might be original user input or one revised by LLM."
    #print(f"*** state: {state}")
    #print(f"*** config: {config}")
    from langchain_openai import ChatOpenAI
    from agent.mongo_agent import run_structured_chat_agent
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    database_schema = get_schema_info.invoke({})

    # context = []
    # from langchain_core.messages import HumanMessage, AIMessage
    # for msg in state["messages"]:
    #     if isinstance(msg, HumanMessage):
    #         context.append(f"Question: {msg.content}")
    #     elif isinstance(msg, AIMessage):
    #         context.append(f"Answer: {msg.content}")

    result = run_structured_chat_agent(user_question, database_schema, llm=llm, tools=mongo_tools)
    return to_json_serializable_doc(result)

# # Creating StructuredTool objects for insertion and extraction functions
# tool_extract = StructuredTool.from_function(heka_database.perform_extraction)
# # ...
# tool_query: BaseTool = tool(heka_database.perform_extraction)
# tool_query.name = "ToolQuery"

db_tools = [
    get_schema_info,
    get_user_info,
    get_msp_info,
    get_customer_info,
    get_invoice_info,
#    mongo_query,
]

all_tools = search_tools + db_tools
