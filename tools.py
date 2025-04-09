from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, PubMedAPIWrapper, ArxivAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import requests
import json
import re
from bs4 import BeautifulSoup
from typing import Optional

# Initialize shared LLM configurations
def get_llm(temperature=0.5):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/quasar-alpha",  
        temperature=temperature
)

# PubMed API Wrapper setup
pubmed = PubMedAPIWrapper(top_k_results=7)

# ArXiv API Wrapper setup
arxiv = ArxivAPIWrapper(top_k_results=5)

# Tool implementations for health timeline scanner
def search_health_news_impl(query: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search for health and medical news between specified dates"""
    search_query = f"health medical news"
    
    # Add query if provided and not general
    if query and query.lower() != "all" and query.lower() != "general":
        search_query = f"{query} {search_query}"
    
    # Add date range if provided
    if start_date and end_date:
        search_query += f" from {start_date} to {end_date}"
    
    # Using DuckDuckGo search as a proxy for news search
    results = DuckDuckGoSearchRun().run(search_query)
    return results

def search_pubmed_impl(query: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search PubMed for medical research papers"""
    # Handle general queries
    if not query or query.lower() in ["all", "general", "latest"]:
        search_query = "medical breakthrough OR health innovation OR new treatment"
    else:
        search_query = query
    
    # Add date constraints if provided
    if start_date and end_date:
        # Format dates for PubMed date range query
        # Remove hyphens for PubMed format
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        search_query += f" AND ({start_clean}[Date - Publication] : {end_clean}[Date - Publication])"
    
    try:
        results = pubmed.run(search_query)
        return results
    except Exception as e:
        return f"Error searching PubMed: {str(e)}. Using fallback search method."

def search_arxiv_impl(query: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search arXiv for recent scientific papers"""
    # Handle general queries
    if not query or query.lower() in ["all", "general", "latest"]:
        search_query = "cat:q-bio OR medicine OR healthcare OR medical"
    else:
        search_query = query
        
    try:
        results = arxiv.run(search_query)
        
        # Manual filtering by date if specified (ArXiv API doesn't support date filtering directly)
        if start_date and end_date:
            filtered_results = ""
            for line in results.split("\n"):
                # Look for date patterns in the results
                date_match = re.search(r'\b\d{4}-\d{2}-\d{2}\b', line)
                if date_match:
                    paper_date = date_match.group(0)
                    if start_date <= paper_date <= end_date:
                        filtered_results += line + "\n"
            
            if filtered_results:
                return filtered_results
            else:
                return f"No arXiv papers found within date range {start_date} to {end_date}. Original results:\n{results}"
        
        return results
    except Exception as e:
        return f"Error searching arXiv: {str(e)}. Using fallback search method."

def search_clinical_trials_impl(condition: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search for clinical trials related to a health condition registered between dates"""
    base_url = "https://clinicaltrials.gov/api/query/study_fields"
    
    # Handle general queries
    if not condition or condition.lower() in ["all", "general", "latest"]:
        condition_query = ""  # Will return recent trials
    else:
        condition_query = condition
    
    # Prepare parameters for ClinicalTrials.gov API
    params = {
        "expr": condition_query,
        "fields": "NCTId,BriefTitle,Condition,StartDate,CompletionDate,LastUpdatePostDate,Phase,StudyType",
        "fmt": "json",
        "min_rnk": 1,
        "max_rnk": 10  # Limiting to 10 results
    }
    
    # Add date filters if provided
    if start_date and end_date:
        # Format for ClinicalTrials.gov: YYYY/MM/DD
        formatted_start = start_date.replace("-", "/")
        formatted_end = end_date.replace("-", "/")
        params["expr"] += f" AND AREA[LastUpdatePostDate]RANGE[{formatted_start},{formatted_end}]"
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            studies = data.get("StudyFieldsResponse", {}).get("StudyFields", [])
            
            if not studies:
                return "No clinical trials found matching the criteria."
            
            # Format results
            result_str = "Clinical Trials Found:\n\n"
            for study in studies:
                result_str += f"ID: {', '.join(study.get('NCTId', ['Unknown']))}\n"
                result_str += f"Title: {', '.join(study.get('BriefTitle', ['Unknown']))}\n"
                result_str += f"Condition: {', '.join(study.get('Condition', ['Unknown']))}\n"
                result_str += f"Phase: {', '.join(study.get('Phase', ['Unknown']))}\n"
                result_str += f"Last Updated: {', '.join(study.get('LastUpdatePostDate', ['Unknown']))}\n\n"
            
            return result_str
        else:
            return f"Error searching clinical trials: Status code {response.status_code}. Using fallback search method."
    except Exception as e:
        search_query = f"{condition} clinical trial registered"
        if start_date and end_date:
            search_query += f" from {start_date} to {end_date}"
        results = DuckDuckGoSearchRun().run(search_query)
        return f"Error accessing ClinicalTrials.gov API: {str(e)}. Using search results instead:\n\n{results}"

def search_fda_approvals_impl(drug_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search for FDA drug or device approvals between specified dates"""
    # Initialize with general query if not specified
    if not drug_type or drug_type.lower() in ["all", "general", "latest"]:
        drug_query = "new drug approval OR medical device approval OR breakthrough therapy"
    else:
        drug_query = f"{drug_type} FDA approval"
    
    # Add date range to search query
    if start_date and end_date:
        date_range = f"from {start_date} to {end_date}"
        search_query = f"{drug_query} {date_range}"
    else:
        search_query = drug_query
    
    # Try to get data from FDA's website directly
    try:
        # This would be replaced with actual FDA API implementation if available
        results = DuckDuckGoSearchRun().run(search_query)
        
        # Enhance the results by extracting from FDA press releases if we had API access
        # For now, return search results
        return results
    except Exception as e:
        return f"Error searching FDA approvals: {str(e)}. Using fallback search method."

def search_health_agencies_impl(query: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search for health guidelines and announcements from major health agencies (CDC, WHO, NIH)"""
    # Handle general queries
    if not query or query.lower() in ["all", "general", "latest"]:
        search_query = "CDC OR WHO OR NIH new guidelines OR health advisory OR medical recommendation"
    else:
        search_query = f"{query} CDC OR WHO OR NIH guidelines OR advisory"
    
    # Add date constraints
    if start_date and end_date:
        search_query += f" from {start_date} to {end_date}"
    
    results = DuckDuckGoSearchRun().run(search_query)
    return results

def search_medical_breakthroughs_impl(query: str = "", start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search specifically for medical breakthroughs and innovations"""
    search_query = "medical breakthrough OR healthcare innovation OR scientific discovery medicine OR new treatment approved"
    
    # Add specific query if provided
    if query and query.lower() not in ["all", "general", "latest"]:
        search_query = f"{query} {search_query}"
    
    # Add date constraints
    if start_date and end_date:
        search_query += f" from {start_date} to {end_date}"
    
    results = DuckDuckGoSearchRun().run(search_query)
    return results

def search_medical_journals_impl(query: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Search medical journals for research published between specified dates"""
    # First try PubMed as the primary source for medical literature
    pubmed_results = search_pubmed_impl(query, start_date, end_date)
    
    # Also try general search as a backup
    search_query = f"medical journal research {query}"
    if start_date and end_date:
        search_query += f" from {start_date} to {end_date}"
    
    general_results = DuckDuckGoSearchRun().run(search_query)
    
    # Combine results
    combined = f"PubMed Results:\n{pubmed_results}\n\nAdditional Results:\n{general_results}"
    return combined

def save_to_txt(data: str, filename: str = "health_timeline.txt") -> str:
    """Save timeline data to a text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Health Timeline Report ---\nGenerated: {timestamp}\n\n{data}\n\n"

    try:
        with open(filename, "w", encoding="utf-8") as f:  # Using 'w' instead of 'a' to overwrite
            f.write(formatted_text)
        
        return f"Timeline successfully saved to {filename}"
    except Exception as e:
        return f"Error saving to file: {str(e)}"

def save_to_html(data: str, filename: str = "health_timeline.html") -> str:
    """Save timeline data to an HTML file with formatting for email sharing"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML with modern styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Health Timeline Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 25px;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-style: italic;
                margin-bottom: 25px;
            }}
            .development {{
                background-color: #f9f9f9;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 0 4px 4px 0;
            }}
            .development-title {{
                font-weight: bold;
                color: #2c3e50;
            }}
            .development-date {{
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .development-category {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                margin-top: 8px;
            }}
            .source {{
                font-size: 0.85em;
                color: #7f8c8d;
                margin-top: 10px;
            }}
            .footer {{
                margin-top: 40px;
                font-size: 0.8em;
                color: #7f8c8d;
                text-align: center;
                border-top: 1px solid #ecf0f1;
                padding-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Health Timeline Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
        
        {data}
        
        <div class="footer">
            Generated by Health Timeline Scanner
        </div>
    </body>
    </html>
    """
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return f"Timeline successfully saved to {filename}"
    except Exception as e:
        return f"Error saving to HTML file: {str(e)}"

def create_engaging_summary(text: str) -> str:
    """Create an engaging, captivating summary of medical developments"""
    llm = get_llm(temperature=0.7)  # Higher temperature for more engaging writing
    
    prompt_template = """
    You are a brilliant science communicator specializing in making complex medical developments exciting and accessible to everyone.
    
    Transform the following medical information into a captivating, clear summary that would engage readers in an email.
    
    Your summary should:
    1. Use vivid language that brings medical breakthroughs to life
    2. Explain complex concepts in simple terms without losing scientific accuracy
    3. Highlight the real-world impact on patients and healthcare
    4. Create a sense of excitement about medical progress
    5. Use analogies where helpful to explain complex medical concepts
    6. Be structured with clear sections and engaging headings
    7. Maintain rigorous scientific accuracy while being accessible
    
    The summary should be comprehensive but concise, perfect for sharing via email.
    
    MEDICAL INFORMATION:
    {text}
    
    CAPTIVATING SUMMARY:
    """
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"text": text})
    
    return response["text"]

def summarize_text(text: str) -> str:
    """Summarize long text content"""
    if len(text) < 500:  # If text is already short, return as is
        return text
        
    doc = Document(page_content=text)
    llm = get_llm(temperature=0.3)  # Lower temperature for factual summary
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary = chain.invoke([doc])
    return summary["output_text"]

def simplify_medical_jargon(text: str) -> str:
    """Convert medical jargon to plain language explanations"""
    llm = get_llm(temperature=0.4)
    
    prompt_template = """
    You are an expert at translating complex medical language into clear, accessible explanations.
    
    Transform the following medical text into simple language that anyone can understand. 
    Keep all the important information but explain:
    - Medical terminology in plain words
    - Complex procedures in simple steps
    - Scientific concepts using everyday analogies
    - Statistics in meaningful context
    
    Original text:
    {text}
    
    Plain language explanation:
    """
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=prompt_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"text": text})
    
    return response["text"]

def deep_reasoning(query: str) -> str:
    """Analyze complex health trends through structured reasoning"""
    llm = get_llm(temperature=0.3)
    
    prompt_template = """
    You are a medical analysis engine designed to analyze health and medical trends through careful reasoning.
    
    QUERY: {query}
    
    Please follow these steps in your analysis:
    1. Break down the medical topic into key components and relevant subtopics
    2. Identify major research directions and clinical significance
    3. Consider different medical perspectives and possible contradicting evidence
    4. Analyze research limitations and potential biases in medical literature
    5. Draw evidence-based conclusions relevant to both clinicians and patients
    6. Summarize implications for healthcare practices and future research
    
    ANALYSIS:
    """
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"query": query})
    
    return response["text"]

def health_impact_analysis(development: str) -> str:
    """Analyze the potential impact of a health development on different populations"""
    llm = get_llm(temperature=0.4)
    
    prompt_template = """
    You are a healthcare impact analyst who specializes in understanding how medical developments affect real people.
    
    For the following health development, provide a thorough analysis of:
    1. Short-term and long-term patient benefits
    2. How this might change clinical practice
    3. Different impacts across demographic groups (age, gender, socioeconomic status)
    4. Economic implications for healthcare systems
    5. Potential challenges to implementation or adoption
    
    HEALTH DEVELOPMENT:
    {development}
    
    IMPACT ANALYSIS:
    """
    
    prompt = PromptTemplate(
        input_variables=["development"],
        template=prompt_template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"development": development})
    
    return response["text"]

# Tool definitions
save_timeline_to_file = Tool(
    name="save_timeline_to_file",
    func=save_to_txt,
    description="Saves health timeline data to a text file."
)

save_timeline_to_html = Tool(
    name="save_timeline_to_html",
    func=save_to_html,
    description="Saves health timeline data to an HTML file formatted for email sharing."
)

search_health_news = Tool(
    name="search_health_news",
    func=search_health_news_impl,
    description="Search for health and medical news between specified dates."
)

search_pubmed = Tool(
    name="search_pubmed",
    func=search_pubmed_impl,
    description="Search PubMed for medical research papers published between specified dates."
)

search_arxiv = Tool(
    name="search_arxiv",
    func=search_arxiv_impl,
    description="Search arXiv for scientific papers on health and medicine topics."
)

search_medical_journals = Tool(
    name="search_medical_journals",
    func=search_medical_journals_impl,
    description="Search medical journals for research published between specified dates."
)

search_clinical_trials = Tool(
    name="search_clinical_trials",
    func=search_clinical_trials_impl,
    description="Search for clinical trials registered between specified dates."
)

search_fda_approvals = Tool(
    name="search_fda_approvals",
    func=search_fda_approvals_impl,
    description="Search for FDA drug or device approvals between specified dates."
)

search_health_agencies = Tool(
    name="search_health_agencies",
    func=search_health_agencies_impl,
    description="Search for guidelines and announcements from major health agencies like CDC, WHO, and NIH."
)

search_medical_breakthroughs = Tool(
    name="search_medical_breakthroughs",
    func=search_medical_breakthroughs_impl,
    description="Search specifically for medical breakthroughs and innovations in a time period."
)

summarize_tool = Tool(
    name="summarize",
    func=summarize_text,
    description="Summarizes long text content into a concise summary."
)

create_engaging_summary_tool = Tool(
    name="create_engaging_summary",
    func=create_engaging_summary,
    description="Transform medical information into captivating, clear summaries perfect for email sharing."
)

simplify_medical_jargon_tool = Tool(
    name="simplify_medical_jargon",
    func=simplify_medical_jargon,
    description="Convert complex medical language into plain, accessible explanations anyone can understand."
)

deep_reasoning_tool = Tool(
    name="deep_reasoning",
    func=deep_reasoning,
    description="Analyzes complex health trends through structured reasoning, examining evidence and providing evidence-based conclusions."
)

health_impact_tool = Tool(
    name="health_impact_analysis",
    func=health_impact_analysis,
    description="Analyze how a health development might impact different populations and healthcare practices."
)

wiki_tool = Tool(
    name="wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)).run,
    description="Search Wikipedia for information about a health or medical topic."
)

# List of all tools for easy import
health_timeline_tools = [
    search_health_news,
    search_pubmed,
    search_arxiv,
    search_clinical_trials,
    search_fda_approvals,
    search_health_agencies,
    search_medical_breakthroughs,
    save_timeline_to_file,
    save_timeline_to_html,
    summarize_tool,
    create_engaging_summary_tool,
    simplify_medical_jargon_tool,
    deep_reasoning_tool,
    health_impact_tool,
    wiki_tool,
    search_medical_journals
]