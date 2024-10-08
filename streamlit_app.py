import asyncio
import os
import json
import uuid
from typing import AsyncGenerator, List
import pprint

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from client import AgentClient
from schema import ChatMessage


# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Heka AI Agent"
APP_ICON = "🧰"
HEKA_LOGO = "https://cdn.grumatic.com/assets/heka/logo-aws-mp.png"

@st.cache_resource
def get_agent_client():
    agent_url = os.getenv("AGENT_URL", "http://localhost")
    return AgentClient(agent_url)


async def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=HEKA_LOGO,
        menu_items={'Get help': 'https://www.grumatic.com',
                    #'About': "Grumatic Heka Solution",
                    },
        layout="centered"
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
          visibility: hidden;
          height: 0%;
          position: fixed;
        }

        img {
          background: white;
        }

        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    models = {
        "OpenAI GPT-4o-mini (streaming)": "gpt-4o-mini",
        "OpenAI GPT-4o (streaming)": "gpt-4o",
        "llama-3.1-70b on Groq": "llama-3.1-70b",
    }
    # Config options
    with st.sidebar:
        st.image(image=HEKA_LOGO)
        #st.header(f"{APP_ICON} {APP_TITLE}")
        st.html("<div style='text-align:center;font-size:1.3em'>Welcome Heka AI Agent</div>")
        with st.popover(":material/settings: Settings", use_container_width=True):
            m = st.radio("LLM to use", options=models.keys())
            model = models[m]
            use_streaming = st.toggle("Stream results", value=True)

#         @st.dialog("Question Samples")
#         def question_samples_dialog():
#             st.caption(
#                 """
# - 헤카란 무엇인가? or what is heka?
# - 그루매틱 회사의 비전은 무엇인가? or what is the vision of Grumatic?
# - 그루매틱의 성장 전력은 무엇인가? or what is the growth strategy of Grumatic company?
# - 헤카 서비스를 사용하는 회사 목록을 알려줘 or what companies are using heka?
# - what organizations are there for company id f3b74b5cc9614419922dd6e0ad074f28
# - can you make a report for the invoice id 20240716009 in terms of cloud cost usage?
# - can you compare details for invoice id 202406623666 and 202407623666 in terms of cost usage?
# """
#             )

#         if st.button("Question Samples", use_container_width=True):
#             question_samples_dialog()

        with st.popover(":material/questions:", use_container_width=True):
            st.json(
                {"Grumatic and Heka Solution":
                 {
                     "Q1": "Who is Grumatic?",
                     "Q2": "What is heka solution?",
                     "Q3": "What is the vision of Grumatic?",
                     "Q4": "What is the growth strategy of Grumatic company?",
                 },
                 "Heka Users, MSPs, Customers":
                 {
                     "Q1": "What MSP companies are in heka?",
                     "Q2": "What users are in heka?",
                     "Q3": "How many customers are in heka?",
                     "Q4": "What customers are managed by Grumatic?",
                     "Q4": "In database, Can you provide a list of all users registered in (주)콤텍시스템 along with their roles?"
                 },
                 "Heka Invoices":
                 {
                     "Q1": "let me know the invoices of 삼프로TV in last month",
                     "Q2": "let me knwo the invoices of 삼프로TV from 2024.1 to 2024.8",
                     "Q3": "In database, let me know the aws invoices of 삼프로TV in last month, limit to 1",
                 }
                 }, expanded=True)

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )
        st.html(
            """
            <div style='text-align:center;'>
              <hr/>
              <div>
               <a href="https://www.heka.so" target="_blank">Heka Home</a>
               &nbsp & &nbsp
               <a href="https://msp.dev.heka.so" target="_blank">Heka MSP</a>
              </div>
              <p></p>
              <p> Powered by Grumatic Inc. </p>
            </div>
            """)


    # Draw existing messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: List[ChatMessage] = st.session_state.messages

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    def new_thread_clicked():
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []

    col1, col2 = st.columns([8,1])
    with col2:
        st.button("New", help="New Thread", on_click=new_thread_clicked)

    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI-powered assistant providing the information about Grumatic company and their Heka solution for cloud cost billing automation and optimization. I may take a few seconds to boot up when you send your first message. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter():
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if input := st.chat_input():
        messages.append(ChatMessage(type="human", content=input))
        st.chat_message("human").write(input)
        agent_client = get_agent_client()
        if use_streaming:
            stream = agent_client.astream(
                message=input,
                model=model,
                #thread_id=get_script_run_ctx().session_id,
                thread_id=st.session_state.thread_id
            )
            await draw_messages(stream, is_new=True)
        else:
            response = await agent_client.ainvoke(
                message=input,
                model=model,
                thread_id=get_script_run_ctx().session_id,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun()  # Clear stale containers

    # If messages have been generated, show feedback widget
    if len(messages) > 0:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new=False,
):
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # matching tool calls to their results
    call_results = {}
    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.json(tool_call["args"], expanded=1)

                        # Expect one ToolMessage for each tool call.
                        # for _ in range(len(call_results)):
                        #     tool_result: ChatMessage = await anext(messages_agen)
                        #     pprint.pp(f"*** {tool_result}\n")
                        #     if not isinstance(tool_result, ChatMessage):
                        #         print("*** tool_result is not ChatMessage type ??? so skip it.")
                        #         continue

                        #     if not tool_result.type == "tool":
                        #         st.error(
                        #             f"Unexpected ChatMessage type: {tool_result.type}"
                        #         )
                        #         st.write(tool_result)
                        #         st.stop()

                        #     # Record the message if it's new, and update the correct
                        #     # status container with the result
                        #     if is_new:
                        #         st.session_state.messages.append(tool_result)
                        #     status = call_results[tool_result.tool_call_id]
                        #     status.write("Output:")
                        #     status.json(tool_result.content, expanded=1)
                        #     status.update(state="complete")

            case "tool":
                # tool message, the result of tool call
                # Record the message if it's new, and update the correct
                # status container with the result
                if is_new:
                    st.session_state.messages.append(msg)

                if not msg.tool_call_id:
                    st.error("Unexpected tool_call_id!")
                    st.write(msg)
                    st.stop()

                status = call_results[msg.tool_call_id]
                status.write("Output:")
                try:
                    json_content = json.loads(msg.content)
                    status.json(msg.content, expanded=1)
                except ValueError:
                    status.write(msg.content)
                status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback():
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback and (latest_run_id, feedback) != st.session_state.last_feedback:

        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client = get_agent_client()
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs=dict(
                comment="In-line human feedback",
            ),
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
