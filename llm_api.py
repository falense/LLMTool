import os
import threading

from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq

# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import RSSFeedLoader
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer

from langchain_core.tools import tool
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers.string import StrOutputParser

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain.globals import set_debug

set_debug(True)

#####################
# Models Definition
#####################

API_VERSION = "2023-12-01-preview"

azureGPT35 = AzureChatOpenAI(
    openai_api_version=API_VERSION,
    azure_deployment="gpt35",
)

azureGPT4 = AzureChatOpenAI(
    openai_api_version=API_VERSION,
    azure_deployment="gpt4",
)

groq_llama70b = ChatGroq(temperature=0, model_name="llama3-70b-8192")
groq_llama8b = ChatGroq(temperature=0, model_name="llama3-8b-8192")

#########################################
# Core prompts
#########################################

system = SystemMessage(
    content="""You are a helpful assistant. Your job is to help the user with any kind of requests and answer them to the best of your ability. You should avoid at all cost to answer with 'I am just an AI'.

    ## Functions

    You have access to a function to retrieve the most recent news headlines. You can use this function to get the most recent news headlines from today. Report the news headlines in the original language. Do not translate the news headlines.

    You have access to a function that can add two numbers together. You can use this function to add two numbers together. Never try to do math yourself, always use the function.
    
    Whenever you receive a tool message this means that you have previously called a tool function. You can use the output of the tool function in your response to the user. Never refer to the tool function itself in your response to the user. Always refer to the output of the tool function.
    
    """
)

system_idea = SystemMessage(
    content="""
    As an adept conversational agent, your role includes facilitating dialogue that reflects a user-centric approach. Your specific task is to craft three (3) potential prompts that echo natural inquiries a user may pose to a chatbot like yourself. These prompts should serve to extend the conversation by anticipating the user's needs or curiosity based on the information exchanged in previous messages.

    During this process, use your comprehension of the context to create prompts that a user might realistically ask, guiding them through their journey of discovery, problem-solving, or general inquiry.

    Follow these detailed steps:
    - Analyze the information shared in the preceding interactions to accurately gauge the user's perspective and possible lines of questioning.
    - Generate three engaging, open-ended questions, adhering closely to the user's conversational style and likely areas of interest.
    - Present each question as if it were formulated by the user, ready for you, the AI chatbot, to answer.
    - Utilize the 'add_prompt_suggestion' function to submit each user-perspective question individually. This compartmentalizes your suggestions, allowing the user to seamlessly select and pursue the topic they find most compelling.

    If the dialogue history does not directly suggest follow-up questions, tap into your creative intellect to propose prompts that reflect common user inquiries applicable to the dialogue's subject matter.

    Your responsibility is not only to simulate a natural conversation but also to anticipate and generate questions that embody the user’s voice and perspective. This proactive engagement showcases your capability to tailor the conversation in a manner that aligns with what users might typically seek from an AI chatbot's assistance.```

    By emphasizing the aim of mirroring a user's potential questions and reiterating the importance of user perspective, this revised prompt aims to better align the AI assistant's outputs with the intent of engaging users in a natural and user-driven conversation.
    """,
)

prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="history"),
        ]
    ) 

#########################################
# Tools Definition for user interaction
#########################################

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

@tool
def find_current_news_headlines() -> str:
    """Access realtime information about current world events and returns the most recent headings for today"""
    try:
        urls = ["https://www.nytimes.com/international/"]
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            docs, tags_to_extract=["p"]
        )

        return docs_transformed[0].page_content
    except:
        return "I'm sorry, I couldn't find the news headlines."


@tool
def find_current_norwegian_news_headlines() -> str:
    """Access realtime information about current events in Norway"""

    try:
        urls = ["https://www.nrk.no/toppsaker.rss"]

        loader = RSSFeedLoader(urls=urls)
        data = loader.load()
        return data[0].page_content
    except:
        return "I'm sorry, I couldn't find the news headlines."

@tool
def find_the_current_forecast_for_city(city: str) -> str:
    """Finds the current weather forecast for a city."""

    try:

        #https://python.langchain.com/docs/use_cases/apis/
        chain = APIChain.from_llm_and_api_docs(
            azureGPT35,
            open_meteo_docs.OPEN_METEO_DOCS,
            verbose=True,
            limit_to_domains=["https://api.open-meteo.com/"],
            return_intermediate_steps=True,
        )

        resp = chain.invoke(
            f"What is the weather like right now in {city} in degrees Celcius?"
        )
        print(resp)

        return resp["output"]
    
    except:
        return "I'm sorry, I couldn't find the weather forecast."

tools = [add, multiply, find_current_news_headlines, find_current_norwegian_news_headlines, find_the_current_forecast_for_city]
functions = [convert_to_openai_function(t) for t in tools]

groq_llama70b_tools = groq_llama70b.bind_tools(tools)
groq_llama8b_tools = groq_llama8b.bind_tools(tools)
azureGPT35_tools = azureGPT35.bind_tools(tools)
azureGPT4_tools = azureGPT4.bind_tools(tools)

def call_tools(response: list) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = response.tool_calls
    tool_messages = []
    #https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/function.py

    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
        tool_messages.append(ToolMessage("Response from AI tool call was: %s" % tool_call["output"], tool_call_id=tool_call["id"]))
    return tool_messages

tool_chain = ( prompt | azureGPT35_tools | {"ai_message": lambda x: x, "tools": call_tools} )
response_chain = ( prompt | azureGPT4  )

#########################################
# Tools Definition for suggestions
#########################################

output_parser = StrOutputParser()
tool_parser = JsonOutputToolsParser()

@tool
def add_prompt_suggestion(prompt: str) -> None:
    """Gives the user a prompt suggestion."""
    return 

tools_ideas = [add_prompt_suggestion]

groq_llama8b_ideas = groq_llama8b.bind_tools(tools_ideas)
groq_llama70b_ideas = groq_llama70b.bind_tools(tools_ideas)
azureGPT35_ideas = azureGPT35.bind_tools(tools_ideas)
azureGPT4_ideas = azureGPT4.bind_tools(tools_ideas)

idea_chain = azureGPT4_ideas | tool_parser

# response = groq_llama70b_tools.invoke([system, human])
# print(response)
