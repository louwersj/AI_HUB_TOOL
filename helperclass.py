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
        You are tasked with analyzing the provided architectural drawings and feasibility documents to deliver a detailed project assessment. The following project details are provided:
            - Project Details: {project_details}
            - Cost Information: {cost_info}
            - Historical Data: {historical_info}
        
        Analysis Instructions:
        1. Elevations (Files Containing Elevation Drawings):
            - Objective: Identify and list materials for each elevation (North, South, East, and West).
            - Methodology:
                1. Use labels and legends within the drawings to determine materials (e.g., brickwork, cladding).
                2. Include the material types and finishes for each elevation.
        
        2. Floor Plan (Files Containing Floor Plans):
            - Objective: Calculate room areas, total floor area, and identify new construction elements.
            - Methodology:
                1. Use the labeled scale ratio and paper size to convert drawing measurements into real-world dimensions.
                2. Measure and calculate the area of each room and the total floor area.
                3. Identify areas marked for work to be done  using the provided PDF :
                    - Calculate the area of each marked region based on the scale and associate it with the corresponding label.
                    - Areas marked with tags (e.g., "allowance for demolition") must be quantified using their highlighted regions, adjusted for the scale, and matched to the corresponding costs in the feasibility estimate.

        
        3. Roof Plan (Files Containing Roof Drawings):
            - Objective: Identify and measure extensions or modifications.
            - Methodology:
                1. Look for extensions (commonly shown as striped side boxes) and calculate their areas using the scale.
        
        4. Feasibility Estimate (Files Containing Cost Data):
            - Objective: Provide an itemized cost breakdown for the project.
            - Methodology:
                1. AI-Driven Costing: For subcategories marked for AI-based costing, generate cost estimates based on the drawings and historical data.
                2. Itemized Breakdown: Include the following major categories with detailed cost estimates:
                    - Demolition
                    - Substructure
                    - Superstructure
                    - Roof
                    - External Windows and Doors
                    - Partitions
                    - Joinery
                    - Electrical
                    - Mechanical
                    - Decoration
                    - Preliminaries
                3. Cost Calculations:
                    - Quantification: Use the drawings and specifications to quantify materials and labor, incorporating BIM (Building Information Modeling) or manual methods.
                    - Unit Rates: Apply rates from cost databases or historical project data.
                    - Summation: Aggregate costs across materials, labor, and equipment to calculate category totals.
        
        
        Final Output:
        Your final response should include the following structured sections:
            1. Elevations:
                - List materials for each elevation (North, South, East, and West).
                - Specify material types, finishes, and any observations from the drawings.
            2. Floor Plan:
                - Provide the scale used.
                - Calculate and report the area of each room and the total floor area.
                - Highlight details of new construction areas (e.g., blue-marked walls).
            3. Roof Plan:
                - Identify and calculate areas of extensions or modifications.
            4. Feasibility Estimate:
                - Include a detailed, itemized cost breakdown for the categories listed.
                - Provide the total project cost, excluding VAT.
        
        Important Notes:
        - Use only the information provided in the drawings and documents; external references are not allowed.
        - Ensure alignment with the project scope and provided cost details.
        - Be precise and methodical in calculations to maintain accuracy.
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
