from crewai import Agent, Task, Crew ,LLM,Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")



search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

llm = LLM(
    model="gpt-4o-mini",
    temperature=0.6,
    max_tokens=4096,
    max_retries=3,
    timeout=10,
)


data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent uses statistical modeling and machine learning to provide crucial insights. With a knack for data, "
              "the Data Analyst Agent is the cornerstone for informing trading decisions.",
    verbose=True,
    llm=llm,
    allow_delegation=True,
    tools = [scrape_tool, search_tool]
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based on insights from the Data Analyst Agent.",
    backstory="Equipped with a deep understanding of financial markets and quantitative analysis, this agent devises and refines trading strategies. "
              "It evaluates the performance of different approaches to determine the most profitable and risk-averse options.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies based on approved trading strategies.",
    backstory="This agent specializes in analyzing the timing, price, and logistical details of potential trades."
              "By evaluating these factors, it provides well-founded suggestions for when and how trades should be executed to maximize efficiency and adherence to strategy.",
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools = [scrape_tool, search_tool]
)

# Task for Data Analyst Agent: Analyze Market Data
data_analysis_task = Task(
    description=( "Continuously monitor and analyze market data for the selected stock ({stock_selection}). "
                  "Use statistical modeling and machine learning to identify trends and predict market movements."),
    expected_output=( "Insights and alerts about significant market opportunities or threats for {stock_selection}." ),
    agent=data_analyst_agent,
)

# Task for Trading Strategy Agent: Develop Trading Strategies
strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on the insights from the Data Analyst and user-defined risk tolerance ({risk_tolerance}). "
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=( "A set of potential trading strategies for {stock_selection} that align with the user's risk tolerance." ),
    agent=trading_strategy_agent,
)

# Task for Trade Advisor Agent: Plan Trade Execution
execution_planning_task = Task(
    description=( "Analyze approved trading strategies to determine the best execution methods for {stock_selection}, considering current market conditions and optimal pricing." ),
    expected_output=("Detailed execution plans suggesting how and when to execute trades for {stock_selection}."),
    agent=execution_agent,
)


financial_trading_inputs = {
    'stock_selection': 'BRK.B',
    # 'initial_capital': '100000',
    'risk_tolerance': 'High',
    'trading_strategy_preference': '1 week hold',
    # 'news_impact_consideration': True
}

# Define the crew with agents and tasks
financial_trading_crew = Crew(
    agents=[data_analyst_agent,trading_strategy_agent,execution_agent], 
           
    
    tasks=[data_analysis_task,strategy_development_task,execution_planning_task  ],
    manager_llm=ChatOpenAI(model="gpt-4o-mini", 
                           temperature=0.7),
    process=Process.hierarchical,
    verbose=True
)
result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)
print(result)