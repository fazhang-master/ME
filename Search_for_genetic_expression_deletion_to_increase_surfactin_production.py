import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["USE_FLASH_ATTENTION"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(r"/home/Qwen1.5-72B-Chat-GPTQ-Int4",
    device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(r"/home/Qwen1.5-72B-Chat-GPTQ-Int4")

#Qwen1.5-72B-Chat-GPTQ-Int4
#Qwen1.5-14B-Chat

from pyvis.network import Network
import pandas as pd
import re
import networkx as nx

# Load the Excel file
filepath = 'modified_updated(Qwen2.5 32b)_Bacillus subtilis lipopeptide or surfactin biosynthetic gene expressionBacillus subtilis lipopeptide or surfactin biosynthetic gene deletion_causal.xlsx'
df = pd.read_excel(filepath, engine='openpyxl')

# Initialize NetworkX Graph
G = nx.Graph()

# Nodes to exclude
words_to_exclude = []

# Regular expression to match the pattern (entity A, entity B)
pattern = r'\(([^,]+), ([^\)]+)\)'

# Iterate over the DataFrame rows to extract entity pairs and their sources
for _, row in df.iterrows():
    value = row['Answer to Question 2']
    source = row['Title']  # Extract source for each pair

    matches = re.findall(pattern, value)
    for entity_a, entity_b in matches:
        # Check if any word to exclude is part of the entity names
        if not any(word in entity_a for word in words_to_exclude) and not any(word in entity_b for word in words_to_exclude):
            G.add_node(entity_a, label=entity_a)
            G.add_node(entity_b, label=entity_b)
            G.add_edge(entity_a, entity_b, title=source)

def search_network(graph, keywords, depth=1):
    # Ensure all keywords are lowercase for case-insensitive search
    keyword_list = [kw.lower() for kw in keywords]

    # Helper function to check if a node label contains all keywords
    def contains_all_keywords(label):
        return all(kw in label.lower() for kw in keyword_list)

    # Collect nodes that contain all keywords in their label
    nodes_of_interest = set()
    for node, attr in graph.nodes(data=True):
        if 'label' in attr and contains_all_keywords(attr['label']):
            nodes_of_interest.add(node)

    # Expand search to include neighbors up to the specified depth
    for _ in range(depth):
        neighbors = set()
        for node in nodes_of_interest:
            neighbors.update(nx.neighbors(graph, node))
        nodes_of_interest.update(neighbors)
    
    # Return a subgraph containing only relevant nodes and edges
    return graph.subgraph(nodes_of_interest).copy()

# Perform search with a list of keywords
word_combinations = ["ethanol","increase"]  # Replace with your keywords
filtered_graph = search_network(G, word_combinations)

# Extract node names from the filtered graph
node_names = list(filtered_graph.nodes())

# Prepare a simple text summary of node names
node_names_text = ", ".join(node_names)

# Now, `node_names_text` contains a clean, comma-separated list of node names, ready for summarization
print(node_names_text)

# Initialize Pyvis network with the filtered graph
net = Network(height="2160px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(filtered_graph)

# Continue with setting options and saving the network as before
net.set_options("""
{
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -80000,
      "centralGravity": 0.5,
      "springLength": 75,
      "springConstant": 0.05,
      "damping": 0.09,
      "avoidOverlap": 0.5
    },
    "maxVelocity": 100,
    "minVelocity": 0.1,
    "solver": "barnesHut",
    "timestep": 0.3,
    "stabilization": {
        "enabled": true,
        "iterations": 500,
        "updateInterval": 10,
        "onlyDynamicEdges": false,
        "fit": true
    }
  },
  "nodes": {
    "font": {
      "size": 30,
      "color": "white"
    }
  }
}
""")

# Save and show the network
net.write_html('filtered_entity_' + "_".join(word_combinations) + '_network.html')

from IPython.display import Markdown

def trim_text(text, max_length):
    if len(text) > max_length:
        return text[:max_length].rsplit(' ', 1)[0] + "..."  # Trim to max_length, avoid cutting words in half
    else:
        return text
    
# Apply the trimming function to node_names_text
cut_off_chunk_size = 5000
trimmed_node_names_text = trim_text(node_names_text, cut_off_chunk_size)
keyword = ", ".join(word_combinations)

# Construct the prompt with the potentially trimmed node_names_text
prompt = "These are the terms related to " + filepath + keyword + ", categorize them and write a summary report.   " + trimmed_node_names_text

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=5000
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
try:
    display(Markdown(response1))
except NameError:
    output_md = "总结报告.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(response1)
    print(f"✅ Markdown文件已保存至: {output_md}")
