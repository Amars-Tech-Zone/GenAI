from langchain_openai import ChatOpenAI
from langgraph_supervisor import supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph_supervisor import create_supervisor



load_dotenv()

model=ChatOpenAI(temperature=0, model="gpt-4o-mini")

def search_duckduckgo(query: str) -> str:
    """Search DuckDuckGo and return the first result."""
   
    search=DuckDuckGoSearchRun()
    return search.invoke(query)

# result=search_duckduckgo("Who is a Mother")
# print(result)

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# math agent
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="MathAgent",
    prompt="An agent that can perform basic math operations like addition and multiplication.",
)

# web agent
search_tool=create_react_agent(
    model=model,
    tools=[search_duckduckgo],
    name="WebSearchAgent",
    prompt="An agent that can search the web using DuckDuckGo."
)

# supervisor created for multiagent
workflow = create_supervisor(
    [search_tool, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

chatbot=workflow.compile()

result = chatbot.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what is quantum computing?"
        }
    ]
})

print(result["messages"])

for m in result["messages"]:
    m.pretty_print()

   
   