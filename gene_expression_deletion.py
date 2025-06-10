print("step1------------------------------------------------------------------------------------------------------")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["USE_FLASH_ATTENTION"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(r"/mnt/zf/Qwen1.5-72B-Chat-GPTQ-Int4",
    device_map="auto", torch_dtype="auto",attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(r"/mnt/zf/Qwen1.5-72B-Chat-GPTQ-Int4")

#Qwen1.5-72B-Chat-GPTQ-Int4
#Qwen1.5-14B-Chat

print("step2------------------------------------------------------------------------------------------------------")
from Bio import Entrez
import pandas as pd

# Define your email to use with NCBI Entrez
Entrez.email = "xiao.zhengyang@wustl.edu"

def search_pubmed(keyword):
    # Adjust the search term to focus on abstracts
    search_term = f"{keyword}[Abstract]"
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=1000)
    record = Entrez.read(handle)
    handle.close()
    # Get the list of Ids returned by the search
    id_list = record["IdList"]
    return id_list

def fetch_details(id_list):
    ids = ','.join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    # Create a list to hold our article details
    articles = []

    for pubmed_article in records['PubmedArticle']:
        article = {}
        article_data = pubmed_article['MedlineCitation']['Article']
        article['Title'] = article_data.get('ArticleTitle')

        # Directly output the abstract
        abstract_text = article_data.get('Abstract', {}).get('AbstractText', [])
        if isinstance(abstract_text, list):
            abstract_text = ' '.join(abstract_text)
        article['Abstract'] = abstract_text

        article['Journal'] = article_data.get('Journal', {}).get('Title')

        articles.append(article)

    return articles

def perform_search_and_fetch(keyword):
    id_list = search_pubmed(keyword)
    return fetch_details(id_list)

# Example usage: Performing two searches
keyword1 = "Bacillus subtilis lipopeptide or surfactin biosynthetic gene expression"
keyword2 = "Bacillus subtilis lipopeptide or surfactin biosynthetic gene deletion"
keyword = keyword1 + keyword2
# Fetch articles for both keywords
articles1 = perform_search_and_fetch(keyword1)
articles2 = perform_search_and_fetch(keyword2)

# Convert both lists of articles to DataFrames
df1 = pd.DataFrame(articles1)
df2 = pd.DataFrame(articles2)

# Add a column to differentiate the search terms in the final DataFrame
df1['SearchTerm'] = keyword1
df2['SearchTerm'] = keyword2

# Concatenate the DataFrames
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to an Excel file
excel_filename = keyword+"_pubmed_search_results.xlsx"
combined_df.to_excel(excel_filename, index=False)

print(f"Saved combined search results to {excel_filename}")

# Qwen reads abstract and identify knowledge

print("step3------------------------------------------------------------------------------------------------------")

import pandas as pd
import os
import torch
import gc

# Assuming `tokenizer`, `model`, and `device` are already defined and initialized
# Example device initialization: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to interact with the LLM API using the new method, now with customizable system prompts
def ask_questions(abstract, questions, system_prompts):
    responses = []
    for question, system_prompt in zip(questions, system_prompts):
        # Combine the question and abstract to form the prompt
        prompt_text = question + " " + str(abstract)
        
        # Prepare the messages for the new API, using a customizable system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Generate response with the new API
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=5000
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses.append(response)
    return responses

# Read the Excel file
file_path = excel_filename  # Replace with your file path
df = pd.read_excel(file_path)

questions = [" "]  # Placeholder question that won't be used

system_prompts = [
    "You are specialized for analyzing scientific paper abstracts, Extract entities and causal relationships from scientific paper abstracts. Focus on genes overexpression/deletion, rationale for expression/deletion, products, and metabolites related to expression/deletion. Output in (expression/deletion of gene xxx, consequence 1), (expression/deletion of gene xxx, consequence 2)... format with no additional text."
]

# Process each abstract and store the response
total_rows = len(df)
for i, row in df.iterrows():
    # Clear the console at the beginning of each iteration
    os.system('cls' if os.name == 'nt' else 'clear')

    # Since we're only asking one question now, directly get the response for the second (index 0) system prompt
    response = ask_questions(row['Abstract'], [questions[0]], [system_prompts[0]])[0]

    # Store the response in the DataFrame
    df.at[i, 'Answer to Question 2'] = response

    # Show the response
    print(f"Response for Row {i+1}:")
    print(f"Answer to Question 2: {response}")

    # Calculate and show the progress percentage
    progress = ((i + 1) / total_rows) * 100
    print(f"total_rows: {total_rows}")
    print(f"Progress: {progress:.2f}% completed")

# Save the updated DataFrame back to an Excel file
output_file_path = 'updated(Qwen2.5 32b)_'+keyword+'_causal.xlsx'  # Replace with your desired output file path
df.to_excel(output_file_path, index=False)

print("step4------------------------------------------------------------------------------------------------------")

# remove repeat words
import pandas as pd
import re

# Load an Excel file
df = pd.read_excel(output_file_path, engine='openpyxl')

# Fill NaN values in 'Response to New Question' column with zero
df['Answer to Question 2'] = df['Answer to Question 2'].fillna(0)

# Convert the column values to strings (to ensure compatibility with re.findall)
column_values = df['Answer to Question 2'].astype(str).tolist()

# Initialize an empty list to hold entities
entities = []

# Regular expression to match the pattern (entity A, entity B)
pattern = r'\(([^,]+), ([^\)]+)\)'

# Iterate over each cell in the column
for value in column_values:
    # Find all matches of the pattern in the cell
    matches = re.findall(pattern, value)
    # For each match, extend the entities list with the extracted entities
    for match in matches:
        entities.extend(match)  # This adds both entity A and entity B to the list

# Remove duplicates if necessary
entities = list(dict.fromkeys(entities))

# Join the entities with commas
entities_string = ', '.join(entities)

print(entities_string)

import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load your Excel file
file_path = output_file_path
df = pd.read_excel(file_path, engine='openpyxl')

# Assuming you have a list of entities and their embeddings already
entities = [entity.strip() for entity in entities_string.split(',')]
t2vmodel = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = t2vmodel.encode(entities)

# Identify similar phrases and store them in a dictionary
similar_phrases = {}

# Calculate total iterations for progress tracking
total_iterations = sum(range(len(entities)))

# Initialize a counter to track progress
current_iteration = 0

for i in range(len(entities)):
    for j in range(i + 1, len(entities)):
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j])
        if similarity.item() > 0.8:
            # Assuming entities[i] is the first phrase and entities[j] is the similar one
            similar_phrases[entities[j]] = entities[i]

        # Update the current iteration counter after each inner loop iteration
        current_iteration += 1

    # Print the percentage completed
    percentage_completed = (current_iteration / total_iterations) * 100
    print(f"Progress: {percentage_completed:.2f}%")

# Note: Printing progress in the inner loop might slow down your code execution,
# especially if 'entities' is very large. You might want to update the progress
# less frequently, for example, only after each completion of the outer loop.


# Specify the column you want to modify
specific_column = 'Answer to Question 2'

# Calculate total iterations for progress tracking (only for the specific column)
total_rows = len(df)
current_iteration = 0

# Iterate through the specific column to substitute similar phrases
for index, row in df.iterrows():
    current_iteration += 1
    # Print progress every 100 rows to avoid performance degradation
    if current_iteration % 100 == 0 or current_iteration == total_rows:
        progress_percentage = (current_iteration / total_rows) * 100
        print(f"Progress: {progress_percentage:.2f}% complete.")

    cell_value = str(row[specific_column])
    for similar, original in similar_phrases.items():
        # Check if the phrase contains 'Yarrowia', if so, skip substitution
        if 'Yarrowia' in cell_value or 'Yarrowia' in similar:
            continue
        if similar in cell_value:
            # Substitute similar phrase with the original phrase, ignoring errors if not found
            try:
                df.at[index, specific_column] = cell_value.replace(similar, original)
            except Exception as e:
                print(f"Error substituting phrase: {e}")
                continue

# Save the modified Excel file
modified_file_path = 'modified_' + file_path
df.to_excel(modified_file_path, index=False, engine='openpyxl')

print("Excel file has been modified and saved as:", modified_file_path)

