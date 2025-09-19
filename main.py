import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,  END
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
import os
import base64
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
    """Use LLM to choose which tool should be called. The user can pass text prompt for text query or image path for image analysis query. SO the LLM should decide which tool to call based on the user query"""
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
            },
            "ImageServer":{
                "command":"python",
                "args":["image_server.py"],
                "transport":"stdio"
            }
        }
    )
    tools=await client.get_tools()
    print("==========",str(tools),"==========")
    agent=create_react_agent(model, tools)
    # if os.path.isfile(state["query"]) and state["query"].lower().endswith((".png", ".jpg", ".jpeg")):
    #     input_msg = f"The user provided an image file path: {state['query']}. Please call the analyze_image tool with this path."
    # else:
    #     input_msg = state["query"]

    input_data = {"messages": [{"role": "user", "content": state['query']}]}
    print("==========Input data===========")
    search_response = await agent.ainvoke(input_data)
    state['tool_response']=search_response['messages'][-1].content
    print("Router node state: ", state['tool_response'])
    return state

# Formatter Node
async def formatter_node(state: GraphState)-> GraphState:
    """Format the final response to the user"""
    query=state['query']
    tool_result=state['tool_response']

    prompt=f"""The user asked: {query}. You got the tool result: {tool_result}. If the tool name is None, apologize to the user for not being able to help.
    Based on this information, provide a descriptive and concise and informative response to the user."""
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
    state_data={'query':query}
    result = await app.ainvoke(state_data)
    print("\n===Final Answer===\n")
    print(result['formatted'])

if __name__ == "__main__":
    asyncio.run(main())