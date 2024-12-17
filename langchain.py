import pandas as pd
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tqdm import tqdm
import time

llm = OpenAI(api_key="#")

prompt_template = """
Extract brand, model, RAM, and ROM from this text:

{text}

Format:
Brand: <brand>
Model: <model>
RAM: <ram>
ROM: <rom>
"""


prompt = PromptTemplate(input_variables=["text"], template=prompt_template)


def extract_phone_details_batch(texts):
    chain = LLMChain(llm=llm, prompt=prompt)
    responses = []
    for text in texts:
        response = chain.run(text)
        responses.append(response)
    return responses


def process_excel_in_chunks(input_file, output_file, chunk_size=30):
    # Read input Excel file
    df = pd.read_excel(input_file)

    # Initialize a new column for extracted details
    df["Phone Details"] = ""

    # Process in chunks
    num_rows = len(df)
    with tqdm(total=num_rows, desc="Processing rows") as pbar:
        for i in range(0, num_rows, chunk_size):
            # Extract texts for the current chunk
            texts_chunk = df.loc[i:i+chunk_size-1, "Name of product"].tolist()

            # Extract phone details for the current chunk
            responses = extract_phone_details_batch(texts_chunk)

            # Update DataFrame with extracted details for the current chunk
            for j, response in enumerate(responses):
                df.at[i + j, "Phone Details"] = response

            # Update progress bar
            pbar.update(chunk_size)

            time.sleep(1)  # Adjust sleep time based on API rate limits

    # Save updated DataFrame back to Excel file
    df.to_excel(output_file, index=False)
# Input and output file paths
input_file = "/Users/bez/Desktop/STR/import data2020.xlsx"
output_file = "/Users/bez/Desktop/STR/output2020.xlsx"

# Process the Excel file in chunks with progress bar
process_excel_in_chunks(input_file, output_file)

print(f"Extraction completed and saved to {output_file}")
