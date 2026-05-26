from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    answer: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-5.4-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that gathers information on a given topic and provides an answer, summary, sources, and tools used.
            Wrap the output in this and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial_format(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt, 
    tools=[]
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=[], 
    verbose=True
)

raw_response = agent_executor.invoke({"query": "What is the capital of France?"})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
except Exception as e:
    print(f"Error parsing response", e, "Raw response: ", raw_response)