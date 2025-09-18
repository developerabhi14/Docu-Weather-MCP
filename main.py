import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,  END
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

load_dotenv()



# Setup mcp clients


# define llm

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.6)

# state for langgraph

class GraphState(dict):
    query: str
    tool_response:str
    formatted:str

# router node (llm based)
async def router_node(state:GraphState)->str:
    """Use LLM to choose which tool should be called"""
    client=MultiServerMCPClient(
        {
            "Weather":{
                "command":"python",
                "args":["weather_api.py"],
                "transport":"stdio"
            },
            'RAG':{
                "command":"python",
                "args":["rag_tool.py"],
                "transport":"stdio"
            }
        }
    )
    tools=await client.get_tools()
    agent=create_react_agent(model, tools)
    search_response=await agent.ainvoke(
        {"messages":[{'role':'user','content':state['query']}]}
    )
    state['tool_response']=search_response['messages'][-1].content
    return state

# Formatter Node
async def formatter_node(state: GraphState)-> GraphState:
    """Format the final response to the user"""
    query=state['query']
    tool_result=state['tool_response']

    prompt=f"""The user asked: {query}. You got the tool result: {tool_result}. If the tool name is None, apologize to the user for not being able to help.
    Based on this information, provide a concise and informative response to the user."""
    response=await model.ainvoke(prompt)
    state['formatted']=response.content.strip()
    return state


graph=StateGraph(GraphState)
graph.add_node("router", router_node)
graph.add_node("formatter", formatter_node)

graph.add_edge(START, "router")
graph.add_edge("router", "formatter")
graph.add_edge("formatter", END)


app=graph.compile()


# run the client
async def main():
    query=input("Enter your query")
    result=await app.ainvoke({'query':query})
    print("\n===Final Answer===\n")
    print(result['formatted'])

if __name__ == "__main__":
    asyncio.run(main())