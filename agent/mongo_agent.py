import dotenv
dotenv.load_dotenv()

# import importlib
# import agent.heka_database
# importlib.reload(agent.heka_database)

import pprint
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain.prompts import PromptTemplate

structured_prompt = hub.pull("hwchase17/structured-chat-agent")
#react_prompt = hub.pull("hwchase17/react")

from langchain_core.exceptions import OutputParserException

def _agent_handle_error(error: OutputParserException) -> str:
    return (
        "The following errors occurred during agent execution: "
        + str(error)
        + " Please finish."
    )

user_prompt_template = PromptTemplate(
    template="""Given the database schema {database_schema}, your task is to perform operations on a NoSQL database for heka cloud billing automation and cost optimization solution. Before proceeding with the query '{user_input}', consider the following relations over collections in database:
    - UID of company collection is COMPANY_ID in other collections
    - UID of organization collection is ORGARNIZATION_ID in other collections
    You can use the context involved in user question: {user_context}""",
    input_variables=["database_schema", "user_input", "user_context"],
)


def run_zero_shot_react_agent(user_input:str,
                              database_schema:dict,
                              llm=None,
                              tools=[],
                              user_context=None,
                              ):
  user_prompt = user_prompt_template.invoke({"database_schema": database_schema,
                                             "user_input": user_input,
                                             "user_context": user_context})
  try:
    # Initializing the agent with the tools and model specified, ready to execute tasks
    agent_executor = initialize_agent(
      tools,  # Tools for the agent to use
      llm,  # The language model to use for processing
      agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # The type of agent to initialize
      verbose=True,  # Enables detailed logging
      handle_parsing_errors=False,
      max_iterations=5,
    )
    # response is str type
    response = agent_executor.run(user_prompt)
    return response
  except Exception as e:
    return f"agent executor error: {e}"

def run_structured_chat_agent(user_input:str,
                              database_schema:dict,
                              llm=None,
                              tools=[],
                              user_context=None):

  user_prompt = user_prompt_template.invoke({"database_schema": database_schema,
                                             "user_input": user_input,
                                             "user_context": user_context})

  try:
    agent = create_structured_chat_agent(llm, tools=tools, prompt=structured_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                   max_iterations=5,
                                   return_intermediate_steps=False,
                                   handle_parsing_errors=False) #_agent_handle_error)
    response = agent_executor.invoke({"input": user_prompt})
    return response["output"]
  except Exception as e:
    # retry with zero_shot_react_agent due to JsonDecode output parsing error
    #return run_zero_shot_react_agent(user_input, database_schema, llm=llm, tools=tools, user_contex=user_context)
    return f"agent executor error: {e}"

# ---- test agent ------
example_queries = [
  "어떤 조직이 있는지 10개만 알려줘"
  # "give me the information about company named (주)콤텍시스템"
  # "let me know the information of user named 예찬호",
  # "let me know the list of users. just limit to 20",
  # "let me know only the name of users. just limit to 3",
  # "let me know only the name of organizations belongs to Grumatic company. just limit to 3",
]

def run_examples(run_agent_func, llm=None, tools=None, database_schema=None):
  if not llm:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

  if not tools:
    from agent.tools import mongo_tools
    tools = mongo_tools # set up the tools

  if not database_schema:
    from agent.tools import get_schema_info
    database_schema = get_schema_info.invoke({})

  for query in example_queries:
    print("---------\n")
    pprint.pp({"input": query,
               "output": run_agent_func(query, database_schema, llm=llm, tools=tools)})

# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field

# class Response(BaseModel):
#     query: list = Field(description="It must be a list")

#     def to_dict(self):
#         return self.query

# parser = PydanticOutputParser(pydantic_object=Response)

# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()

# from langchain.pydantic_v1 import BaseModel, Field
# from langchain_core.tools import StructuredTool, ToolException
# # Define the CalculatorInput schema
# class PerformExtractionInput(BaseModel):
#     collection_name: str = Field(description="collection name for heka mongo database")
#     query: dict = Field(decription="query for mongo database", default={})
#     projection: dict = Field(description="projection for mongo database", default={})
#     limit: int = Field(description="limiting to the number of results", default=30)

# # Custom error handler
# def _handle_error(error: ToolException) -> str:
#     return (
#         "The following errors occurred during tool execution: "
#         + error.args[0]
#         + " Please try another tool."
#     )


# # Creating StructuredTool objects for insertion and extraction functions
# tool_query = StructuredTool.from_function(func=heka_database.perform_extraction,
#                                           name="ToolQuery",
#                                           description="useful to query heka-related data from mongo database",
#                                           return_direct=True,
#                                           args_schema=PerformExtractionInput,
#                                           handle_tool_error=_handle_error,
#                                           )
