import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def multiply(a: int, b: int) -> int:
    """
    Multiply a and b.

    Args:
        a: first int
        b: second int
    """

    return a*b

def add(a: int, b: int) -> int:
    """
    adds a and b.

    Args:
        a: first int
        b: second int
    """

    return a+b

def divide(a: int, b: int) -> float:
    """
    Divide a and b.

    Args:
        a: first int
        b: second int
    """

    return a/b

# Search Tool
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

tools = [add, multiply, divide, search]

llm_with_tools = llm.bind_tools(tools)

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.")

def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display_json

# Graph
builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

# Add Edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
    # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "reasoner")

react_graph = builder.compile()

# messages = [HumanMessage(content = "What's sum of the age of D.Trump and Modi")]
# messages = react_graph.invoke({"messages": messages})

# for m in messages["messages"]:
#     m.pretty_print()


# 1. Define the inputs (the starting state)
inputs = {"messages": [HumanMessage(content="What's difference b/w the age of D.Trump and Modi")]}

# 2. Run the stream
# 'stream_mode="messages"' yields (message_chunk, metadata)
for message, metadata in react_graph.stream(inputs, stream_mode="messages"):
    
    # FILTER 1: Only look at tokens from the 'reasoner' node
    if metadata["langgraph_node"] == "reasoner":
        
        # FILTER 2: Skip tool calls (we only want the final text answer)
        if hasattr(message, "tool_calls") and message.tool_calls:
            continue
            
        # Print tokens as they arrive
        if message.content:
            print(message.content, end="", flush=True)
