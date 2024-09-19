import os
import pprint
from typing import TypedDict, Literal, Annotated, Union
from datetime import datetime
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    ToolMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agent.llama_guard import llama_guard, LlamaGuardOutput

from langchain.globals import set_llm_cache, set_debug
from langchain.cache import SQLiteCache

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from agent.tools import all_tools, get_schema_info, mongo_aggregate, retriever
from agent import heka_database

# for debugging langgraph
#set_debug(True)

# cache LLM interactions, added by xroger88
set_llm_cache(SQLiteCache(database_path="cache.db"))

# for debugging
# for name, value in os.environ.items():
#     print("*** {0}: {1}\n".format(name, value))


# def add_dicts(left, right):
#     # coerce to list
#     if not isinstance(left, list):
#         left = [left]
#     if not isinstance(right, list):
#         right = [right]
#     # coerce to message
#     return (left + right)

import operator
class AgentState(MessagesState):
    user_question: str
    parsed_question: dict
    unique_names: list[str]
    generated_query: dict
    query_result: Union[list[dict], dict, str]
    query_result_relevance: str
    # -----
    query_history: Annotated[list[dict], operator.add]
    summary: str
    safety: LlamaGuardOutput
    #is_last_step: IsLastStep

# from langchain.memory import ConversationSummaryMemory
# from langchain.chains import ConversationChain
# def create_ChatOpenAI(model="gpt-4o-mini", max_retries=0):
#     memory = ConversationSummaryMemory(
#         llm=ChatOpenAI(model=model, temperature=0), return_messages=True
#     )
#     return ChatOpenAI(
#         model=model,
#         temperature=0,
#         streaming=True,
#         max_retries=max_retries,
#     )


# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models = {
    # NOTE: gpt4o-mini-2024-07-18 and gpt-4o-2024-08-06 support structured output with json_format as output_format
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True),
    # NOTE: max_retrys=0 to reduce gpt-4o rate limit error
    "gpt-4o": ChatOpenAI(
        model="gpt-4o-2024-08-06", temperature=0, streaming=True, max_retries=0, max_tokens=1024
    ),
    # NOTE: Korean not fully supported, sometimes LLM internal error occurred
    "llama-3.1-70b": ChatGroq(model="llama-3.1-70b-versatile", temperature=0),
}


# NOTE: this is for LangGraph Studio Input Configuration
# Define the config
class GraphConfig(TypedDict):
    model: Literal[
        # "llama3", "mistral", "groq", "anthropic",
        "gpt-4o-mini",
        "gpt-4o",
        "llama-3.1-70b",
    ]
    recursion_limit: int


current_date = datetime.now().strftime("%B %d, %Y")

def get_remove_messages(messages: list):
    "메시지 히스토리에서 ToolMessage 메시지 및 AIMessage 메지지 중에서 tool_calls가 있는 경우 제거하기 위한 RemoveMessage 리스트 출력"
    remove_messages = []
    for msg in messages:
        # remove ToolMessage to reduce the tokens of chat history
        if isinstance(msg, ToolMessage) or (
            isinstance(msg, AIMessage) and msg.tool_calls
        ):
            remove_messages.append(RemoveMessage(id=msg.id))
    return remove_messages


def call_chat_history(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    remove_messages = get_remove_messages(messages)
    pprint.PrettyPrinter(indent=4, width=80).pprint(
        f"*** state: {state}\nremove_messages:{remove_messages}"
    )
    return {"messages": remove_messages}


def call_summarize_conversation(state: AgentState, config: RunnableConfig):
    summary = state.get("summary", "")
    # skip the summarization if the number of messges is less than 6
    if len(state["messages"]) < 7:
        return {"summary": summary}

    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    # do summarize the conversation
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# from agent import mongo_agent
# def call_mongo_agent(state: AgentState, config: RunnableConfig):
#     llm = models[config["configurable"].get("model", "gpt-4o-mini")]
#     last_message = state["messages"][-1]
#     response = mongo_agent.run(llm, last_message.content)
#     return {"query_history": [{'query': last_message.content,
#                                'result': str(response)}]}

def parse_question(state: AgentState, config: RunnableConfig):
    question = state["messages"][-1].content
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a data analyst that can help summarize NOSQL document collctions and parse user questions about a database.
Given the question and database schema, identify the relevant collections and attributes.
If the question is not relevant to the database or if there is not enough information to answer the question, set is_relevant to false.

# A few things to remember about database schema:
    - Company is a Managed Service Provider (MSP) that provides cloud management service for their customers.
    - Organization is a customer that are companies specially using cloud management service.
    - There are Cloud Service Providers (CSP) such as 'aws', 'gcp', 'azure', 'nhn', 'ncloud', etc. Each customer can use cloud computing resources provided by multiple CSPs.
    - Invoice is the billing information of cloud cost usage monthly issued to organizations (customers).

# Use the following synonym dictionary for interpreting user question into the terms of database schema:
    {{
      "user": ["사용자", "유저"],
      "company": ["회사", "MSP", "Managed Service Provider"],
      "organization": ["고객", "고객사", "Customer"],
      "invoice": ["청구", "청구서", "bill"],
    }}

Your response should be in the following JSON format:
{{
    "is_relevant": boolean,
    "names": [string],
    "relevant_collections": [
        {{
            "collection_name": string,
            "attributes": [string],
            "name_attributes": [string]
        }}
    ]
}}

If the question includes names for user or company or organization, fill up the names in the response format. Do not convert name, for example, "(주)주식회사" to "주)주식회사".

The "name_attributes" field should contain only the attributes that are relevant to the question and contain "name" word in their attribute name, for example, the attribute "first_name" in "user" collection contains names relevant to the question "what users are there?", but the attributes like "uid", "address", "business_location" are not relevant because it does not contain "name".
'''),
        ("human", "===Database schema:\n{schema}\n\n===User question:\n{question}\n\nIdentify relevant collections and attributes:")
        ])

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    chain = prompt | llm | JsonOutputParser()
    response = chain.invoke({"question": question,
                             "schema": get_schema_info.invoke({}),
                             })

    for name in response["names"]:
        try:
            user_info_list = heka_database.get_user_info(id=name)
            response["user_info"] = user_info_list[0]
        except:
            try:
                msp_info_list = heka_database.get_company_info(id=name)
                response["msp_info"] = msp_info_list[0]
            except:
                try:
                    customer_info_list = heka_database.get_organization_info(id=name)
                    response["customer_info"] = customer_info_list[0]
                except:
                    continue

    return {"user_question": question, "parsed_question": response}


def get_unique_names(state: AgentState):
    """Find unique names in relevant collections and attributes."""

    # FIX
    return {"unique_names": []}

    parsed_question = state['parsed_question']
    if not parsed_question['is_relevant']:
        return {"unique_names": []}

    unique_names = set()
    for coll_info in parsed_question['relevant_collections']:
        coll_name = coll_info['collection_name']
        name_attributes = coll_info['name_attributes']

        if name_attributes:
            projection = {}
            for attr in name_attributes:
                projection[attr] = 1
            results = heka_database.find(coll_name,
                                         query={},
                                         projection={"_id": 0 } | projection,
                                         limit= 0)
            print(results)
            for doc in results:
                unique_names.update(str(value) for key, value in doc.items() if value)

    return {"unique_names": list(unique_names)}


class GenerateQueryResponse(BaseModel):
    collection_name: str = Field(description="collection name for query")
    query: list = Field(description="It must be a list")

    def to_dict(self):
        return {"collection_name": self.collection_name,
                "query": self.query}

generate_query_output_parser = PydanticOutputParser(pydantic_object=GenerateQueryResponse)


def generate_query(state: AgentState, config: RunnableConfig):
    question = state["user_question"]
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]

    def populate_partial_variables() -> dict:
        return {"format_instructions": generate_query_output_parser.get_format_instructions(),
                "current_date": current_date}


    parsed_question = state["parsed_question"]
    # use match statement and inside match don't use `id` instead of `id` use "user_id" with the value of "{user_id}"
    prompt = PromptTemplate(
        template = """
        Create a MongoDB raw aggregation pipeline for the following user question:
        ###{question}###

        Today's date is {current_date}.

        This is relevant database schema : ${db_schema}$

        ### Here is question-related context information ###
        User: ${user_info}$
        Company: ${msp_info}$
        Organization: ${customer_info}$

        ### add multiple instruction based on your requirenment
        DO not use 'refresh_tokens', 'password' in user collection for security purpose.
        Do not use preamble and explanation in json output

        Just return the [] of aggregration pipeline.
        The following is the format instructions.
        ***{format_instructions}***
        """,
        input_variables=["question", "db_schema", "schema_description",
                         "user_info", "msp_info", "customer_info"],
        partial_variables=populate_partial_variables(),
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    chain = prompt | llm | JsonOutputParser()
    relevant_schema = [{item["collection_name"]: item["attributes"]} for item in parsed_question["relevant_collections"]]
    response = chain.invoke({"question": question,
                             "db_schema": relevant_schema,
                             "user_info": parsed_question.get("user_info", None),
                             "msp_info": parsed_question.get("msp_info", None),
                             "customer_info": parsed_question.get("customer_info", None),
                             })
    return {"generated_query": response}


def execute_query(state: AgentState, config: RunnableConfig):
    question = state["user_question"]
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]

    generated_query = state["generated_query"]
    query_result = mongo_aggregate.invoke({"collection_name": generated_query["collection_name"],
                                           "pipeline": generated_query["query"]})

    prompt = PromptTemplate(
        template="""You are a validator for database query result. \n
        Here is the query result: \n\n {query_result} \n\n
        Give a binary score 'yes' or 'no' to indicate whether the query result is valid. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation. \n
        If the query result is empty or [], grade it as not valid. \n
        If the query result contains 'error' word, grade it as not valid. \n
        """,
        input_variables=["question", "query_result"],
    )
    grader = prompt | llm | JsonOutputParser()
    relevance = grader.invoke({"question": question, "query_result": query_result})

    return {"query_result": query_result,
            "query_result_relevance": relevance["score"]}


def answer_with_query_result(state: AgentState, config: RunnableConfig):
    question = state["user_question"]
    llm = models[config["configurable"].get("model", "gpt-4o-mini")]

    query_result = state["query_result"]

    prompt = PromptTemplate(
        template="""You are a helpful assistant in answering user question based on the database query result. \n
        This is about Heka database query result. \n

        Here is the user question: {question} \n
        Here is the query result: \n\n {query_result} \n\n

        First, describe the query result for user question in markdown format.
        Second, answer the question by interpreting the query result.
        Third, guide user to rewrite question in order to get better query result if result is not enough to answer.
        """,
        input_variables=["question", "query_result"],
    )
    chain = prompt | llm
    response = chain.invoke({"question": question, "query_result": query_result})
    return {"messages": [AIMessage(content=response.content)]}

def route_question(question):
    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to a vectorstore or web search.
        Use the vectostore for questions on Grumatic company and its vision, Heka solution and its product features and customer support.
        #Grumatic's products: CostClipper, PayerPro, Heka MSP, Heka FinOps.
        You do not need to be stringent with the keywords in the question related to these topics.
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
        Return the a JSON with a single key 'datasource' and no premable or explanation.
        Question to route: {question}""",
        input_variables=["question"],
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    router = prompt | llm | JsonOutputParser()
    return router.invoke({"question": question})

def grade_documents(question):
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    grader = prompt | llm | JsonOutputParser()
    docs = retriever.invoke(question)
    doc_txt = [doc.page_content for doc in docs[:3]]
    return grader.invoke({"question": question, "document": doc_txt})


    # Take a step by step approach to make an anwser for the questions.
    # Do consider the query history given in the context before using retriever or web search
    # Do explain rationale about why you choose such an answer.

    # - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
    #   so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".

instruction_prompt = PromptTemplate(
    template="""
    You are a helpful assistant as an expert on heka product service which is
    a kind of cloud cost billing automation and optimization solution provided by Grumatic company.
    You can find some data via database query, document retrieval, web search, etc.
    You can provide useful insights about how to optimize the cloud cost.

    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    """,
    input_variables=["current_date"],
    )

def modify_state(state: AgentState):
    summary = state.get("summary", "")
    summary_message = f'#Summary of conversation earlier: {summary}.\nUse the summariezd text for answering the questions.'
    query_history = state.get("query_history", [])
    query_history_message = f'#Query History: {query_history}.\nUse the query history for answering the questions.'

    instructions = instruction_prompt.invoke({"current_date": current_date}).text

    if summary and query_history:
        system_message = f"{instructions}\n\n {summary} \n\n {query_history_message}"
    elif summary:
        system_message = f"{instructions}\n\n {summary_message}"
    elif query_history:
        system_message = f"{instructions}\n\n {query_history_message}"
    else:
        system_message = f"{instructions}"

    return ([SystemMessage(content=system_message)] + state["messages"])

def wrap_model(model: BaseChatModel):
    model = model.bind_tools(all_tools)
    preprocessor = RunnableLambda(
        lambda state: modify_state(state),
        name="StateModifier",
    )
    # FIX: .with_retry seems not working ???
    return preprocessor | model.with_retry(retry_if_exception_type=(RateLimitError,))


async def acall_agent(state: AgentState, config: RunnableConfig):
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    pprint.PrettyPrinter(indent=4, width=80).pprint(state["messages"])
    # TODO: RateLimitError handing needed, exponential backoff!
    response = await model_runnable.ainvoke(state, config)
    # if state["is_last_step"] and response.tool_calls:
    #     return {
    #         "messages": [
    #             AIMessage(
    #                 id=response.id,
    #                 content="Sorry, need more steps to process this request.",
    #             )
    #         ]
    #     }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig):
    safety_output = await llama_guard("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig):
    safety: LlamaGuardOutput = state["safety"]
    output_messages = []

    # Remove the last message if it's an AI message
    last_message = state["messages"][-1]
    if last_message.type == "ai":
        output_messages.append(RemoveMessage(id=last_message.id))

    content_warning = f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    output_messages.append(AIMessage(content=content_warning))
    return {"messages": output_messages}


from langgraph.prebuilt import ToolInvocation, ToolExecutor
tool_executor = ToolExecutor(all_tools)

# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation for each tool call
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"],
        )
        tool_invocations.append(action)

    # We call the tool_executor and get back a response
    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    # We use the response to create tool messages
    tool_messages = [
        ToolMessage(
            content=str(response),
            name=tc["name"],
            tool_call_id=tc["id"],
        )
        for tc, response in zip(last_message.tool_calls, responses)
    ]

    # We return a list, because this will get added to the existing list
    return {"messages": tool_messages}

# Define the graph
agent = StateGraph(AgentState, config_schema=GraphConfig)
# agent.add_node("chat_history", call_chat_history)
# agent.add_node("mongo_agent", call_mongo_agent)
# agent.add_node("summarize_conversation", call_summarize_conversation)

agent.add_node("parse_question", parse_question)
agent.add_node("get_unique_names", get_unique_names)
agent.add_node("generate_query", generate_query)
agent.add_node("execute_query", execute_query)
agent.add_node("answer_with_query_result", answer_with_query_result)
agent.add_node("agent", acall_agent)
from langgraph.pregel import RetryPolicy
agent.add_node("tools", ToolNode(all_tools), retry=RetryPolicy(max_attempts=3))
# agent.add_node("tools", call_tool)


# agent.add_node("guard_input", llama_guard_input)
# agent.add_node("block_unsafe_content", block_unsafe_content)
# agent.set_entry_point("guard_input")

# agent.set_entry_point("summarize_conversation")
# agent.add_edge("summarize_conversation", "model")

# agent.set_entry_point("mongo_agent")
# agent.add_edge("mongo_agent", "model")
agent.set_entry_point("parse_question")

# # We now define the logic for determining whether to end or summarize the conversation
# def should_summarize(state: State) -> Literal["summarize_conversation", END]:
#     """Return the next node to execute."""
#     messages = state["messages"]
#     # If there are more than six messages, then we summarize the conversationㅠ
#     if len(messages) > 6:
#         return "summarize_conversation"
#     # Otherwise we can just end
#     return END


# Check for unsafe input and block further processing if found
# def unsafe_input(state: AgentState):
#     safety: LlamaGuardOutput = state["safety"]
#     match safety.safety_assessment:
#         case "unsafe":
#             return "block_unsafe_content"
#         case _:
#             return "model"
# agent.add_conditional_edges(
#     "guard_input",
#     unsafe_input,
#     {"block_unsafe_content": "block_unsafe_content", "model": "model"}
# )

# Always END after blocking unsafe content
# agent.add_edge("block_unsafe_content", END)


def checking_database_relevance(state: AgentState) -> Literal["yes", "no"]:
    parse_question = state["parsed_question"]
    if parse_question["is_relevant"]:
        return "yes"
    else:
        return "no"

agent.add_conditional_edges("parse_question",
                            checking_database_relevance,
                            {"yes": "get_unique_names",
                             "no": "agent"})

agent.add_edge("get_unique_names", "generate_query")
agent.add_edge("generate_query", "execute_query")

def checking_query_result_relevance(state: AgentState) -> Literal["yes", "no"]:
    relevance = state["query_result_relevance"]
    if relevance == "yes":
        return "yes"
    else:
        return "no"

agent.add_conditional_edges("execute_query",
                            checking_query_result_relevance,
                            {"yes": "answer_with_query_result",
                             "no": "agent"})

agent.add_edge("answer_with_query_result", END)

# Always run "model" after "tools"
agent.add_edge("tools", "agent")

# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return END

agent.add_conditional_edges(
    "agent",
    pending_tool_calls,
    {"tools": "tools",
     END: END}
)

heka_agent = agent.compile(
    checkpointer=MemorySaver(),
)


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}
        result = await heka_agent.ainvoke(
            inputs,
            config=RunnableConfig(configurable={"thread_id": uuid4()}),
        )
        result["messages"][-1].pretty_print()

        # Draw the agent graph as png
        # requires:
        # brew install graphviz
        # export CFLAGS="-I $(brew --prefix graphviz)/include"
        # export LDFLAGS="-L $(brew --prefix graphviz)/lib"
        # pip install pygraphviz
        #
        # heka_agent.get_graph().draw_png("agent_diagram.png")

    asyncio.run(main())
