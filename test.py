import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew,LLM
from crewai.flow.flow import Flow, listen, start
from crewai_tools import WebsiteSearchTool

# Importing AI Suite for adhoc LLM calls and Pydantic
from pydantic import BaseModel
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = LLM(
    model="gpt-4o-mini",
    temperature=0,
)


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

research_agent = Agent(
    role="You are a helpful assistant that can answer questions about the web.",
    goal="Answer the user's question.",
    backstory="You have access to a vast knowledge base of information from the web.",
    tools=[
      WebsiteSearchTool(website=urls[0]),
      WebsiteSearchTool(website=urls[1]),
      WebsiteSearchTool(website=urls[2]),
    ],
    llm=llm,
)

task = Task(
  description="Answer the following question: {question}",
  expected_output="A detailed and accurate answer to the user's question.",
  agent=research_agent,
)

crew = Crew(
    agents=[research_agent],
    tasks=[task],
)
result = crew.kickoff({"question": "What does Lilian Weng say about the types of agent memory"})

print(result)