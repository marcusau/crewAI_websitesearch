import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew,LLM
from crewai.flow.flow import Flow, listen, start
from crewai_tools import WebsiteSearchTool

# Importing AI Suite for adhoc LLM calls and Pydantic
from pydantic import BaseModel
from openai import OpenAI
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
########################################################
class QAState(BaseModel):
  """
  State for the documentation flow
  """
  question: str = "What does Lilian Weng say about the types of agent memory?"
  improved_question: str = ""
  answer: str = ""
  
class QAFlow(Flow[QAState]):
    
  @start()
  def rewrite_question(self):
    print(f"# Rewriting question: {self.state.question}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {
          "role": "system",
          "content": f"""Look at the input and try to reason about the underlying semantic intent / meaning.
            Here is the initial question:
            -------
            {self.state.question}
            -------
            Formulate an improved question:"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.8
    )

    print(f"# Improved question: {response.choices[0].message.content}")

    improved_question = response.choices[0].message.content
    self.state.improved_question = improved_question


  @listen(rewrite_question)
  def answer_question(self):
    print(f"# Answering question: {self.state.improved_question}")
    result = crew.kickoff(inputs={'question': self.state.improved_question})
    self.state.answer = result.raw
    return result

flow = QAFlow()
flow.plot()

# Display the flow visualization using HTML
from IPython.display import IFrame
IFrame(src='crewai_flow.html', width='100%', height=600)
result = flow.kickoff()
print("=" * 10)
print(result)