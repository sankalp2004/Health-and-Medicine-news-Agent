from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chains import LLMChain
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from datetime import datetime, timedelta
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
import os
import re
import time
import pdfkit
import json

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Import tools from the tools.py file
from tools import health_timeline_tools, create_engaging_summary_tool

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

class HealthDevelopment(BaseModel):
    date: str = Field(description="Date of the development in YYYY-MM-DD format")
    title: str = Field(description="Brief title of the development")
    description: str = Field(description="Detailed description of the health/medical development")
    impact: str = Field(description="Real-world impact or significance of this development")
    source: str = Field(description="Source of the information")
    category: str = Field(description="Category (e.g., Research, FDA Approval, Clinical Trial, Treatment, Policy)")

class TimelineSummary(BaseModel):
    time_period: str = Field(description="The time period that was analyzed")
    key_findings: str = Field(description="Overall summary of key health and medical developments")
    major_trends: List[str] = Field(description="List of major trends identified in this period")
    notable_developments: List[HealthDevelopment] = Field(description="List of notable developments in chronological order")
    patient_impact: str = Field(description="How these developments might impact patients and healthcare delivery")
    future_outlook: str = Field(description="Brief outlook on future directions based on these developments")
    tools_used: List[str] = Field(description="List of tools used to gather this information")

class ChatResponse(BaseModel):
    message: str = Field(description="Response message to the user")
    sources: List[str] = []
    tools_used: List[str] = []

def get_llm(temperature=0.5):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/quasar-alpha",
        temperature=temperature
    )

llm = get_llm(temperature=0.5)

timeline_parser = PydanticOutputParser(pydantic_object=TimelineSummary)

timeline_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are a specialized Health Timeline Scanner that compiles and analyzes health and medical developments over specified time periods.
         Your goal is to identify major trends, breakthroughs, and developments that impact healthcare, while organizing this information into a clear, comprehensive timeline.
         Search across multiple sources including medical journals, news, clinical trials, FDA approvals, and health agency guidelines.
         Focus on accuracy, completeness, and providing context about the significance of each development.
         
         Provide output in JSON format using this structure:{format_instructions}
         """),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=timeline_parser.get_format_instructions())

timeline_agent = create_tool_calling_agent(
    llm=llm,
    prompt=timeline_prompt,
    tools=health_timeline_tools
)

timeline_executor = AgentExecutor(agent=timeline_agent, tools=health_timeline_tools, verbose=True)

chat_parser = PydanticOutputParser(pydantic_object=ChatResponse)
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a Medical Timeline Assistant that helps users understand medical and health developments.
            You can search for recent research, clinical trials, FDA approvals, and health news across specific time periods.
            Provide clear, accurate information with references to the sources when available.
            
            When responding conversationally, use this structure:{format_instructions}
            """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=chat_parser.get_format_instructions())

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=1000)

chat_agent = create_tool_calling_agent(
    llm=llm,
    prompt=chat_prompt,
    tools=health_timeline_tools
)

chat_executor = AgentExecutor(
    agent=chat_agent,
    tools=health_timeline_tools,
    memory=memory,
    verbose=True
)

def parse_time_period(period_text):
    today = datetime.now()
    period_text = period_text.lower()

    year_range_match = re.search(r'(\d{4})\s*[-–—to]*\s*(\d{4})', period_text)
    single_year_match = re.search(r'\b(20\d{2})\b', period_text)

    if year_range_match:
        start_year = int(year_range_match.group(1))
        end_year = int(year_range_match.group(2))
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif single_year_match:
        year = int(single_year_match.group(1))
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "last week" in period_text:
        end_date = today
        start_date = today - timedelta(days=7)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "last month" in period_text:
        end_date = today
        start_date = today - timedelta(days=30)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "last year" in period_text:
        end_date = today
        start_date = today - timedelta(days=365)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    elif "this year" in period_text:
        start_date = datetime(today.year, 1, 1)
        end_date = today
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    # Default to last 3 months if time period can't be determined
    end_date = today
    start_date = today - timedelta(days=90)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def generate_email_html_from_summary(summary: dict) -> str:
    return f"""
    <html>
    <head>
        <meta charset='UTF-8'>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; padding: 30px; color: #333; line-height: 1.6; }}
            h1 {{ color: #0a74da; }}
            h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            .section {{ margin-bottom: 30px; }}
            .development {{ margin-bottom: 20px; }}
            a {{ color: #0a74da; text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>Health and Medical Timeline: {summary['time_period']}</h1>

        <div class='section'>
            <h2>🧠 Key Findings</h2>
            <p>{summary['key_findings']}</p>
        </div>

        <div class='section'>
            <h2>📈 Major Trends</h2>
            <ul>
                {''.join(f"<li>{trend}</li>" for trend in summary['major_trends'])}
            </ul>
        </div>

        <div class='section'>
            <h2>🌟 Notable Developments</h2>
            {''.join(f"""
            <div class='development'>
                <strong>{dev['date']}: {dev['title']}</strong>
                <p>{dev['description']}</p>
                <p><em>Impact:</em> {dev['impact']}</p>
                <p><strong>Category:</strong> {dev['category']}</p>
            </div>
            """ for dev in summary['notable_developments'])}
        </div>

        <div class='section'>
            <h2>💡 Patient Impact</h2>
            <p>{summary['patient_impact']}</p>
        </div>

        <div class='section'>
            <h2>🔮 Future Outlook</h2>
            <p>{summary['future_outlook']}</p>
        </div>

        <div class='section'>
            <h2>📚 Sources</h2>
            <ul>
                <li>Wikipedia</li>
                <li>DuckDuckGo</li>
                <li>PubMed</li>
                <li>ArXiv</li>
                <li>ClinicalTrials.gov</li>
            </ul>
        </div>
    </body>
    </html>
    """


def save_summary_pdf(summary, filename):
    html = generate_email_html_from_summary(summary)
    
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    pdfkit.from_string(html, filename, configuration=config)




def main():
    print("🩺 Health Timeline Assistant")
    user_input = input("Enter the time period you want to analyze (e.g., 'last year', '2023-2024', 'this year'): ")

    start_date, end_date = parse_time_period(user_input)
    print(f"\n🔎 Analyzing developments from {start_date} to {end_date}...\n")

    prompt = f"Generate a health and medical development timeline between {start_date} and {end_date}."

    try:
        result = timeline_executor.invoke({"input": prompt})
        print("🧪 Raw result:", result)  # for debugging

        # Extract and parse output
        if isinstance(result, dict) and "output" in result:
            try:
                summary = json.loads(result["output"])
            except json.JSONDecodeError:
                print("⚠️ Failed to parse JSON from output:")
                print(result["output"])
                return
        elif isinstance(result, dict) and all(k in result for k in ["time_period", "key_findings"]):
            summary = result
        elif isinstance(result, str):
            print("⚠️ Got a string response instead of structured output:")
            print(result)
            return
        else:
            print("⚠️ Unexpected result format:", result)
            return

        # Save report
        html = generate_email_html_from_summary(summary)
        with open("health_summary.html", "w", encoding="utf-8") as f:
            f.write(html)
        save_summary_pdf(summary, "health_summary.pdf")

        print("✅ Report saved as 'health_summary.html' and 'health_summary.pdf'.")

    except Exception as e:
        print("❌ Error while generating timeline:", e)

if __name__ == "__main__":
    main()
