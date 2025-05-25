from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage



load_dotenv()

llm=ChatOpenAI(temperature=0,model="gpt-4o-mini")


api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

tools=[wiki_tool]
llm_tools_bindings=llm.bind_tools(tools)

# result=llm_tools_bindings.invoke("Hello world!")

agent_executor = create_react_agent(llm, tools)

response = agent_executor.invoke({"messages": [HumanMessage(content="what is agentic ai")]})


print(response["messages"])

print(response["messages"][-1].content)



