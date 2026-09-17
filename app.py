import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

# Load environment variables
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("groq_api_key", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_api_key", "")
os.environ["WEATHERSTACK_API_KEY"] = os.getenv("WEATHERSTACK_api_key", "")

# Streamlit page
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Single AI Agent")
st.write("Ask me questions about current information or weather.")

# Tavily Search Tool
search_tool = TavilySearch(max_results=3)


# Weather Tool
@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "WEATHERSTACK_API_KEY is not set."

    url = (
        "http://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data for {city}: {data}"

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )

    except Exception as e:
        return f"Weather error: {str(e)}"


# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"]
)

# Tools
tools = [
    search_tool,
    get_weather
]

# Create Agent
agent = create_agent(
    model=llm,
    tools=tools
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask something...")

if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })

                answer = response["messages"][-1].content

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# Sidebar
with st.sidebar:
    st.header("🛠️ Tools")
    st.write("🔎 Tavily Search")
    st.write("🌤️ WeatherStack")
    st.write("🧠 Groq LLM")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()