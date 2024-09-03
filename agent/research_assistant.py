import os
from typing import TypedDict, Literal
from datetime import datetime
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agent.tools import tools
from agent.llama_guard import llama_guard, LlamaGuardOutput

from langchain.globals import set_llm_cache, set_debug
from langchain.cache import SQLiteCache

set_debug(True)
# cache LLM interactions, added by xroger88
set_llm_cache(SQLiteCache(database_path="cache.db"))


class AgentState(MessagesState):
    safety: LlamaGuardOutput
    is_last_step: IsLastStep


# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
models = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True),
    # NOTE: max_retrys=0 to reduce gpt-4o rate limit error
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0, streaming=True, max_retries=0),
    # NOTE: Korean not fully supported, sometimes LLM internal error
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


current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful assistant as an expert on heka product service which is a kind of cloud cost billing automation and optimization solution provided by Grumatic company.
    Using tools for querying invoice data related to heka solution from heka database, you can analyze the data and provide useful insights about how to optimize the cloud cost.
    For other user questions not related to heka, you have the ability to search the web for information.
    Today's date is {current_date}.
    
    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


def wrap_model(model: BaseChatModel):
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    # FIX: .with_retry seems not working ???
    return preprocessor | model.with_retry(retry_if_exception_type=(RateLimitError,))


async def acall_model(state: AgentState, config: RunnableConfig):
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    model_runnable = wrap_model(m)
    # TODO: RateLimitError handing needed, exponential backoff!
    response = await model_runnable.ainvoke(state, config)
    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
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


# Define the graph
agent = StateGraph(AgentState, config_schema=GraphConfig)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
# agent.add_node("guard_input", llama_guard_input)
# agent.add_node("block_unsafe_content", block_unsafe_content)
# agent.set_entry_point("guard_input")
agent.set_entry_point("model")

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

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", END: END})

research_assistant = agent.compile(
    checkpointer=MemorySaver(),
)


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        inputs = {"messages": [("user", "Find me a recipe for chocolate chip cookies")]}
        result = await research_assistant.ainvoke(
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
        # researcH_assistant.get_graph().draw_png("agent_diagram.png")

    asyncio.run(main())
