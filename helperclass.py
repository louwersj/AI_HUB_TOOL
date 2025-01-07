from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from openai import ChatCompletion
import pandas as pd
import os


def load_historical_data():
    """
    Loads and processes historical data from a CSV file.
    Returns:
        dict: Processed historical data categorized by work type and sub-category.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'Final_Work_Categories_and_Price_Analysis.csv')
    df = pd.read_csv(csv_path)

    historical_data = {}
    for _, row in df.iterrows():
        category = row['category']
        sub_category = row['sub-category']
        unit = row['Unit']
        avg_price = row['average_price']
        range_price = row['range_price']

        if category not in historical_data:
            historical_data[category] = []

        historical_data[category].append({
            'sub_category': sub_category,
            'unit': unit,
            'avg_price': avg_price,
            'range_price': range_price
        })

    return historical_data


def load_pdf_contents(pdf_paths):
    """
    Loads contents from a list of PDF files.
    Args:
        pdf_paths (list): List of paths to PDF files.
    Returns:
        str: Concatenated content of all PDF files.
    """
    documents_content = ""
    for pdf in pdf_paths:
        try:
            loader = PyPDFLoader(pdf)
            document = loader.load()
            for page in document:
                documents_content += page.page_content + "\n"
        except Exception as e:
            print(f"Error loading PDF {pdf}: {e}")
    return documents_content


def setup_llm(api_key):
    """
    Sets up the language model with the provided API key.
    Args:
        api_key (str): OpenAI API key.
    Returns:
        OpenAI: Configured language model.
    """
    return OpenAI(model='gpt-4o-2024-05-13', openai_api_key=api_key)


def extract_project_info(payload):
    """
    Extracts project information from a given payload.
    Args:
        payload (list): List of dictionaries containing project details.
    Returns:
        dict: Extracted project information.
    """
    project_info = {}
    for item in payload:
        question = item["question"]
        answer = item["answer"]
        if "Project Type" in question:
            project_info["work_type"] = answer
        elif "gross internal area" in question:
            project_info["gross_internal_area"] = answer
        elif "expected finishes and materials" in question:
            project_info["expected_finishes"] = answer
        elif "site-specific conditions" in question:
            project_info["site_conditions"] = answer
    return project_info


def extract_cost_info(payload):
    """
    Extracts cost information from a given payload.
    Args:
        payload (list): List of dictionaries containing cost details.
    Returns:
        dict: Extracted cost information.
    """
    cost_info = {}
    for section in payload:
        section_name = section["name"]
        items = section["generic"] + section["specific"]
        cost_info[section_name] = [
            {
                "title": item["title"],
                "isChecked": item["isChecked"],
                "value": item.get("value"),
                "quantity": item.get("quantity"),
                "rate": item.get("rate")
            }
            for item in items if item["isChecked"]
        ]
    return cost_info


def formulate_question(project_details, cost_info, historical_data):
    """
    Formulates a question for the language model based on project and cost details.
    Args:
        project_details (dict): Dictionary containing project details.
        cost_info (dict): Dictionary containing cost details.
        historical_data (dict): Dictionary containing historical cost data.
    Returns:
        str: Formulated question.
    """
    historical_info = ""
    for category, items in historical_data.items():
        historical_info += f"{category}:\n"
        for item in items:
            historical_info += (f"  Sub-category: {item['sub_category']}\n"
                                f"  Unit: {item['unit']}\n"
                                f"  Average Price: {item['avg_price']}\n"
                                f"  Range Price: {item['range_price']}\n")

    question = f"""
   Project Overview:
You are provided with the following project details:
    Project Details: {project_details}
    Cost Information: {cost_info}
    Historical Data: {historical_info}
Your task is to analyze the provided architectural drawings and documents, following these specific instructions:
1. Elevations (Files Containing Elevations)
    Identify and list the materials used for each elevation (North, South, East, and West).
    Use the labels beside each material name to ensure accuracy.
2. Floor Plan (Files Containing Floor Plans)
    Locate the floor measurements on the drawing.
    Use the label beside "scale" to determine the real-life to image scale ratio, followed by the paper size.
    Calculate the area of each room, the total area of the floor plan, and identify any new construction, particularly those indicated by blue-colored walls.
3. Roof Plan (Files Containing Roof Plans)
    Detect any extensions (represented as side boxes covered in stripes) and calculate their area.
4. Feasibility Estimate (Files Containing Feasibility Estimate)
    Based on the project's cost details and historical data:
        AI-Driven Costing: If a subcategory mentions using AI, estimate the cost of the required work based on the provided drawings.
        Generate a detailed, itemized cost breakdown for the new project, considering average prices from past projects.
        The breakdown should cover all major categories, including but not limited to:
            Demolition
            Substructure
            Superstructure
            Roof
            External Windows and Doors
            Partitions
            Electrical
            Mechanical
            Preliminaries
Cost Calculations Methodology:
    Quantification: Use construction drawings, specifications, and BIM (Building Information Modeling) software to quantify materials and labor.
    Unit Rates: Apply unit rates (cost per unit of measure) from cost databases or historical data.
    Summation: Sum the costs for each category (materials, labor, equipment) to arrive at the total cost.
    Contingencies: Include a percentage for contingencies to cover unforeseen costs.
    Overheads and Profits: Add a percentage for overheads and profits.
Final Output:
Your final response should include the following sections:
    Elevations: Detail the materials used for North, South, East, and West elevations.
    Floor Plan: Provide the scale, area calculations for each room, total area, and details of any new construction (highlighted in blue).
    Roof Plan: Detail any extensions identified in the roof plan.
    Feasibility Estimate: Provide a cost estimate summary, including a breakdown of costs for demolition, substructure, superstructure, etc., and a total excluding VAT.
Important Notes:
    Do not use external sources; base your analysis solely on the provided information.
    Follow the project details and cost Information carefully as they outline the scope and specifics of the work to be done.
    """
    return question


def chat_completion(messages, api_key):
    """
    Generates a chat completion using the provided messages and API key.
    Args:
        messages (list): List of messages for the chat completion.
        api_key (str): OpenAI API key.
    Returns:
        str: Response from the language model.
    """
    response = ChatCompletion.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        api_key=api_key
    )
    return response['choices'][0]['message']['content']
