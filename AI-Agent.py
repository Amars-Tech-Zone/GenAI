from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch the current weather data for a given city using the WeatherAPI.
    
    Args:
        city (str): The name of the city to fetch weather data for.
    
    Returns:
        str: The weather data in JSON format as a string.
    """
    url = f'https://api.weatherapi.com/v1/current.json?key=7f89b2f47cae43ccbf9170047252105&q={city}&aqi=no'
    response = requests.get(url)
    return response.json()

llm = ChatOpenAI()
prompt = hub.pull("hwchase17/react")
# prompt = hub.pull("rlm/rag-prompt")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

response = agent_executor.invoke({"input": "Find the capital of Odisha, then find it's current weather condition"})
print(response)
response['output']
