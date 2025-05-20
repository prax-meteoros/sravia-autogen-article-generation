
# Standard libraries for OS, JSON, UUIDs, async, threading, and typing
import os
import json
import uuid
import asyncio
import urllib.parse
from threading import Thread
from typing import List



# Flask for API creation
from flask import Flask, request, jsonify

# Typing extensions and data validation
from typing_extensions import Annotated
from pydantic import BaseModel

import requests
import threading
# AutoGen core components for agent setup and messaging
import autogen_core
from autogen_core import (
    DefaultTopicId, MessageContext, SingleThreadedAgentRuntime, TopicId,
    TypeSubscription, message_handler, RoutedAgent, CancellationToken
)

# Message and model handling in AutoGen
from autogen_core.models import (
    SystemMessage, UserMessage, AssistantMessage, ChatCompletionClient,
    LLMMessage, FunctionExecutionResult, FunctionExecutionResultMessage
)

# Tool wrapper and Azure OpenAI client
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Rich output for CLI rendering
from rich.console import Console
from rich.markdown import Markdown

# Set environment variables for Azure OpenAI configurationos.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-mini"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-mini"          # Azure model deployment name
os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"             # API version for Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = "6qZqBQ5RB7ImYsP7ajBcIX7rZUnD22vTz76QI5R3FWlWAJ0EjvcXJQQJ99BEACHYHv6XJ3w3AAAAACOGdXrz"                             # Your Azure OpenAI API key (fill this)
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ai-hpesgalite4096ai784823033224.openai.azure.com"                            # Azure OpenAI endpoint URL (fill this)

# Initialize the Azure OpenAI chat model client
model_client = AzureOpenAIChatCompletionClient(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],              # Deployment name
    api_base=os.environ["AZURE_OPENAI_ENDPOINT"],                  # Endpoint URL
    api_type="azure",                                              # API type
    api_version=os.environ["OPENAI_API_VERSION"],                  # API version
    api_key=os.environ["AZURE_OPENAI_API_KEY"]                     # API key
)



# Initialize the Flask web application
app = Flask(__name__)


# Async function to generate an article title using LLM
async def generate_title(topic: Annotated[str, "The user-provided article topic"]) -> str:
    prompt = f"""You are a headline expert. Generate a professional, concise, and catchy article title for the topic below.

    Topic: {topic}

    Only return the title text. Do not include quotes, labels, or extra commentary."""

    # Call the LLM to get a title based on the prompt
    completion = await model_client.create(
        messages=[UserMessage(content=prompt, source="user")]
    )
    return completion.content.strip()  # 🧹 Return clean title text

# 🛠 Register the function as a tool for use in AutoGen workflows
generate_title_tool = FunctionTool(
    generate_title,
    description="Use LLM to generate a professional article title based on the input topic."
)


# Load the article template from file
with open("article_template.txt", "r") as f:
    article_template = f.read()

# Initialize Rich console for pretty terminal output
console = Console()



# Define a message wrapper for group chat communication
class GroupChatMessage(BaseModel):
    body: UserMessage  # Holds a user-generated message

# Define a signal model to request speaking in group chat
class RequestToSpeak(BaseModel):
    pass  # Used as a trigger with no additional data

# Define a base group chat agent using AutoGen's RoutedAgent
class BaseGroupChatAgent(RoutedAgent):
    def __init__(self, description: str, group_chat_topic_type: str, model_client: ChatCompletionClient, system_message: str) -> None:
        super().__init__(description=description)
        self._group_chat_topic_type = group_chat_topic_type  # Group topic category/type
        self._model_client = model_client                    # Azure/OpenAI model client
        self._system_message = SystemMessage(content=system_message)  # Initial persona setup
        self._chat_history: List[LLMMessage] = []            # Track conversation history

    # Handle incoming user messages routed to this agent
    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.extend([
            UserMessage(content=f"Transferred to {message.body.source}", source="system"),  # System note
            message.body  # Add the actual user message
        ])

    # Handle a request to speak event (e.g., when the agent is chosen to respond)
    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        console.print(Markdown(f"### {self.id.type}: "))  # Print who is speaking
        # Add a system instruction to assume persona
        self._chat_history.append(UserMessage(content=f"Transferred to {self.id.type}, adopt the persona immediately.", source="system"))

        # Query the model with system message + history
        completion = await self._model_client.create([self._system_message] + self._chat_history)
        assert isinstance(completion.content, str)

        # Store and display assistant's response
        self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))
        console.print(Markdown(completion.content))

        # Publish the assistant's message to the group chat
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=completion.content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type)
        )

# Agent responsible for generating article title and content
class ContentCreatorAgent(BaseGroupChatAgent):
    def __init__(self, group_chat_topic_type: str, model_client: ChatCompletionClient):
        # Initialize with a system message and markdown template instruction
        super().__init__(
            "Content Creator",  # Agent description
            group_chat_topic_type,
            model_client,
            f"""You are a content creator. First, use the `generate_title` tool to come up with a professional article title. Then write a full article using the provided markdown structure:\n{article_template}"""
        )
        self._tools = [generate_title_tool]  # Register the title generation tool

    # 🎙️ Respond when asked to speak in the group chat
    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        console.print(Markdown(f"### {self.id.type}: "))  # Print agent role visibly
        self._chat_history.append(
            UserMessage(content="Generate the article starting with title generation.", source="system")
        )

        # 🔄 First attempt: let LLM decide to generate or invoke a tool
        completion = await self._model_client.create(
            messages=[self._system_message] + self._chat_history,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token
        )

        if isinstance(completion.content, str):
            # 🧾 LLM directly responded with article content
            self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))
        else:
            # 🔧 Tool call path: LLM invoked the title generation tool
            tool_call = completion.content[0]
            arguments = json.loads(tool_call.arguments)
            tool_result = await self._tools[0].run_json(arguments, ctx.cancellation_token)

            print(f"🏷️ Generated Title: {tool_result.strip()}")

            # Add the tool call and result to the chat history
            self._chat_history.append(
                AssistantMessage(content=[tool_call], source=self.id.type)
            )
            self._chat_history.append(
                FunctionExecutionResultMessage(content=[
                    FunctionExecutionResult(
                        call_id=tool_call.id,
                        content=tool_result,
                        is_error=False,
                        name=self._tools[0].name
                    )
                ])
            )

            # Re-run LLM after tool execution to generate article body
            completion = await self._model_client.create(
                messages=[self._system_message] + self._chat_history,
                cancellation_token=ctx.cancellation_token
            )
            self._chat_history.append(AssistantMessage(content=completion.content, source=self.id.type))

        # Display the final article content
        console.print(Markdown(self._chat_history[-1].content))

        # Broadcast the final message to the group chat
        await self.publish_message(
            GroupChatMessage(body=UserMessage(content=self._chat_history[-1].content, source=self.id.type)),
            topic_id=DefaultTopicId(type=self._group_chat_topic_type)
        )

# Agent to format content using the predefined article markdown template
class FormatterAgent(BaseGroupChatAgent):
    def __init__(self, group_chat_topic_type: str, model_client: ChatCompletionClient):
        super().__init__(
            "Formatter",  # Agent role/description
            group_chat_topic_type,
            model_client,
            f"You are a formatter. Format the content into the expected markdown layout using the following template:\n{article_template}"
        )

# Agent to check if the article formatting follows the expected markdown structure
class FormatCheckerAgent(BaseGroupChatAgent):
    def __init__(self, group_chat_topic_type: str, model_client: ChatCompletionClient):
        super().__init__(
            "Format Checker",  # Agent role/description
            group_chat_topic_type,
            model_client,
            "You are a format checker. Verify if the article is correctly formatted as per the template. Respond only with APPROVED or specific SUGGESTIONS."
        )



# GroupChatManager orchestrates the agent conversation flow
class GroupChatManager(RoutedAgent):
    def __init__(self, participant_topic_types: List[str], model_client: ChatCompletionClient, participant_descriptions: List[str], topic: str) -> None:
        super().__init__("Group Chat Manager")  # Agent name
        self._participant_topic_types = participant_topic_types        # List of agent types (e.g., ContentCreator, Formatter)
        self._model_client = model_client                              # LLM client
        self._chat_history: List[UserMessage] = []                     # Tracks user messages in the chat
        self._participant_descriptions = participant_descriptions      # Descriptions for each agent
        self._previous_participant_topic_type: str | None = None       # Prevent immediate repetition
        self._topic = topic                                            # Topic of the article
        self._last_complete_article = ""                               # Stores article before approval

    # Handles incoming messages from agents
    @message_handler
    async def handle_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self._chat_history.append(message.body)

        if isinstance(message.body.content, str):
            content = message.body.content.strip()
            print(f"🔹 Received content from {message.body.source}: {content[:60]}...")

            # Store completed article before final approval
            if content.endswith("End of the article."):
                self._last_complete_article = content
                print("✅ Stored complete article content in memory.")

            # If format checker approves, save the article to file
            elif content == "APPROVED" and self._last_complete_article:
                base_filename = self._topic.strip() or "article"
                with open(f"./output/{base_filename}.md", "w") as f:
                    f.write(self._last_complete_article)
                print(f"📄 Saved approved article to '{base_filename}.md'.")

                # 🔚 Stop runtime gracefully (if set)
                if self._runtime:
                    print("🛑 Stopping runtime after approval.")
                    # self._runtime.request_termination()  # Optional: explicitly stop if runtime support exists
                else:
                    print("⚠ Runtime reference is not set; cannot stop runtime.")
                return

        # Decide next speaking agent based on chat history
        history = "\n".join([f"{m.source}: {m.content}" for m in self._chat_history])
        roles = "\n".join([
            f"{t}: {d}" for t, d in zip(self._participant_topic_types, self._participant_descriptions)
            if t != self._previous_participant_topic_type  # Avoid back-to-back same agent
        ])

        prompt = (
            "You are managing a collaborative article generation task.\n"
            f"{roles}\n\nConversation:\n{history}\n\n"
            "Select the next role to speak (respond ONLY with one of: ContentCreator, Formatter, FormatChecker)."
        )

        system_message = SystemMessage(content=prompt)

        # Ask model to choose next agent
        completion = await self._model_client.create([system_message], cancellation_token=ctx.cancellation_token)
        response = completion.content.strip()
        print(f"🔍 Model suggested next role: '{response}'")

        # Validate and trigger the next speaker
        selected_topic_type = next(
            (t for t in self._participant_topic_types if t.lower() == response.lower()), None
        )

        if selected_topic_type:
            self._previous_participant_topic_type = selected_topic_type
            await self.publish_message(RequestToSpeak(), DefaultTopicId(type=selected_topic_type))
            print(f"➡ Published RequestToSpeak to: {selected_topic_type}")
        else:
            print(f"❌ Invalid role selected by model: '{response}' — skipping this round.")



# Flask route to trigger article generation via HTTP POST
@app.route("/generate-article", methods=["POST"])
def generate_article():
    # Extract the article topic from query params or fallback to default
    topic = request.args.get("topic") or request.args.get("T") or "Benefit of AI in Healthcare"
    topic = topic.strip()

    # Event to block response until background generation finishes
    import threading
    done_event = threading.Event()

    # Launch group chat in a background thread
    def run_generation():
        asyncio.run(_run_group_chat(topic))  # Run async function inside thread
        done_event.set()  # Notify completion

    thread = Thread(target=run_generation)
    thread.start()

    # ⏱Wait until the generation thread completes
    done_event.wait()

    filename = f"{topic}.md"
    return jsonify({
        "status": "done",
        "message": "Article generated",
        "file": filename,
        "topic": topic
    })

# Asynchronous orchestration of agents for article generation
async def _run_group_chat(topic):
    runtime = SingleThreadedAgentRuntime()  # Single-threaded AutoGen runtime
    participants = ["ContentCreator", "Formatter", "FormatChecker"]  # Agent types
    descriptions = [
        "Creates initial article content.",
        "Formats the article.",
        "Checks article formatting."
    ]

    # Register each agent in the runtime
    await ContentCreatorAgent.register(runtime, "ContentCreator", lambda: ContentCreatorAgent("group_chat", model_client))
    await FormatterAgent.register(runtime, "Formatter", lambda: FormatterAgent("group_chat", model_client))
    await FormatCheckerAgent.register(runtime, "FormatChecker", lambda: FormatCheckerAgent("group_chat", model_client))
    await GroupChatManager.register(runtime, "GroupChatManager", lambda: GroupChatManager(participants, model_client, descriptions, topic))

    # Subscribe each agent to relevant topic types for message routing
    for t in participants:
        await runtime.add_subscription(TypeSubscription(topic_type=t, agent_type=t))           # Direct topic
        await runtime.add_subscription(TypeSubscription(topic_type="group_chat", agent_type=t)) # Shared group chat
    await runtime.add_subscription(TypeSubscription(topic_type="group_chat", agent_type="GroupChatManager"))

    runtime.start()  # Start message loop
    session_id = str(uuid.uuid4())  # Generate unique session

    # Trigger initial article generation instruction
    await runtime.publish_message(
        GroupChatMessage(
            body=UserMessage(
                content=f"Write a professional article on '{topic}' using the given template.",
                source="User",
            )
        ),
        TopicId(type="group_chat", source=session_id),
    )

    await runtime.stop_when_idle()  # Stop when all agents are idle
    await model_client.close()      # Clean up model client connection


# Function to launch Flask server
def run_flask():
    print("Starting Flask server at http://localhost:5002 ...")
    app.run(port=5002, debug=False, use_reloader=False)  # Start Flask app on port 5001

if __name__ == "__main__":
    # Run Flask app in a background thread so we can do other work concurrently
    run_flask()
