import os
from groq import Groq
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings

import json

FILENAME = 'data/un_regulations_157_graph.json'
USERNAME = 'neo4j'
URI = 'neo4j://localhost:7687'
MODEL_NAME = 'llama3-70b-8192'

NODE_SELECTION = """Goal:Identify the most relevant node.
Context:
The graph database contains a list of automotive regulation compliance text, each node represents a requirement or a definition.
Each node has a 'text' attribute which contains the actual regulation.
The relationships represent either a hierarchical relationship (node 3.2.1 is child_of 3.2) or a mention relation (node 4.2.5.1 mentions regulation 2.1.0)
Input:
- question: User's query
- list_of_nodes: A dict of relevant nodes w.r.t. user query, where the key is the node_id and the value node_text refers to the description of the node
Task: Your task is to identify the node thats most relevant for the input query.
Instructions:
- From the input nodes identify one node thats most relevant for the user's query (called n1) by analysing the relevance of the text to the query
- After having conducted the analysis, report only the end result and not the breakdown of the analysis itself.
- Return the id of the node in a yaml format with key 'output'
Note:
- Do not return anything else except the yaml output as the result shall be used by an automated code.
Example:
```
Question:
    "What are the procedures to take to ensure mechanical safety?"
List of nodes:
{{
    '4.3.1': "A transition demand shall not endanger the safety of the vehicle occupants or other road users.",
    '9.3.1': "Most guidelines relating to vehicle inspection and safety are referred to in 3.7.2",
    '3.5.4.2': "Procedures manual for a safe sotware updates are included below. This does not include mechanical activations"
}}
// it can be seen that '9.3.1' is the most relevant to the user's query
Output: '9.3.1'
```
OK, now here's the input
Question:
{query}
List_of_nodes:
{list_of_nodes}
Output:
"""

QA_PROMPT_TEMPLATE = """Task: Formulate an extensive answer to the question given the returned context and the user's query.
The context data contains a automotive compliance regulations, most of the time its either a requirement or a definition.
The regulations given in the context are identified by their id located at the beginning and followed by a colon (3.2.1.:, 2.4.: etc...). Include regulation id in your answer whenever necessary.
Use the following pieces of context to answer the query at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:
"""

QA_GENERATION_PROMPT = PromptTemplate(input_variables=["context", "question"], template=QA_PROMPT_TEMPLATE)
with open(FILENAME) as f:
    data = json.load(f)
all_nodes = [node["id"] for node in data]
graph = Neo4jGraph(url=URI, username=USERNAME, password=os.environ.get("NEO4J_PWD"))
llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
    api_key=os.environ.get('OPEN_AI_KEY')
)

# create graph
'''
driver = GraphDatabase.driver(URI, auth=(USERNAME, os.environ.get("NEO4J_PWD")))
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

for index, node in enumerate(data):
    node_embeddings = embeddings_model.embed_query(node["text"])
    create_node_query = """
    CREATE (n:Node {id: $id, text: $text, embeddings: $embeddings})
    RETURN n
    """
    graph.query(create_node_query, {"id": node["id"], "text": node["text"], "embeddings": node_embeddings})

all_relations = []
for node in data:
    all_relations += node["relations"]
# Create Relationships
for index, relation in enumerate(all_relations):
    src, relationship, dst = relation
    if dst + '.' in all_nodes:
        dst += '.'
    if dst not in all_nodes:
        continue
    if relationship == "mentions":
        create_relation_query = f"""
        MATCH (a:Node {{id: '{src}'}}), (b:Node {{id: '{dst}'}})
        CREATE (a)-[:mentions]->(b)
        """
    else:
        create_relation_query = f"""
        MATCH (a:Node {{id: '{src}'}}), (b:Node {{id: '{dst}'}})
        CREATE (a)-[:child_of]->(b)
        """
    graph.query(create_relation_query)
'''
graph.refresh_schema()
print(graph.schema)


question = "Which requirements should be met for the vehicle type submitted for approval?"
# identify most relevant node
def get_similar_nodes(question, top_k):
    fetch_top_nodes="""WITH $query_embedding AS queryEmbedding
    MATCH (n:Node)
    WITH n, 
        reduce(dot = 0.0, i IN range(0, size(n.embeddings) - 1) | dot + n.embeddings[i] * queryEmbedding[i]) AS dot_product,
        reduce(norm1 = 0.0, i IN range(0, size(n.embeddings) - 1) | norm1 + n.embeddings[i] ^ 2) AS norm1,
        reduce(norm2 = 0.0, i IN range(0, size(queryEmbedding) - 1) | norm2 + queryEmbedding[i] ^ 2) AS norm2
    WITH n, dot_product / (sqrt(norm1) * sqrt(norm2)) AS similarity
    WHERE similarity > 0.0 // Optional: Filter negative similarities if desired
    ORDER BY similarity DESC
    RETURN n.id AS id, n.text as text, similarity as sim
    LIMIT $top_k;
    """
    query_embedding = embeddings_model.embed_query(question)
    similar_nodes = graph.query(fetch_top_nodes, {"query_embedding": query_embedding, "top_k":top_k})
    print(f'Similar nodes to the query {[(result["id"], result["sim"]) for result in similar_nodes]}')
    return {result['id']: result['text'] for result in similar_nodes}

def get_most_relevant_node(question, similar_nodes_dict):
    query = NODE_SELECTION.format(query=question, list_of_nodes=similar_nodes_dict)
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        temperature=0.8,
        model=MODEL_NAME,
    )
    completion = chat_completion.choices[0].message.content
    most_relevant_node = ''
    list_tokens = completion.split(' ')
    for token in list_tokens:
        token = token.replace('"', '')
        token = token.replace("'", "")
        if token in all_nodes:
            most_relevant_node = token
            break
    if most_relevant_node == '':
        print('LLM failed to return a valid node')
        most_relevant_node = list(similar_nodes_dict.keys())[0]
    most_relevant_txt = similar_nodes_dict[most_relevant_node]
    print(f'Most relevant node {most_relevant_node}: {most_relevant_txt}')
    return most_relevant_node, most_relevant_txt

similar_nodes_dict = get_similar_nodes(question)
most_relevant_node, most_relevant_txt = get_most_relevant_node(question, similar_nodes_dict)

# build context
def build_context(most_relevant_node, most_relevant_txt):
    fetch_related_nodes = f"""
    MATCH (n:Node {{id: '{most_relevant_node}'}})-[:mentions]->(mentionedNode:Node)
    RETURN mentionedNode.id AS id, mentionedNode.text AS text;
    """
    related_nodes = graph.query(fetch_related_nodes)
    print(f'Related nodes to the query {related_nodes}')

    # answer the query
    full_context = [most_relevant_node + ': ' + most_relevant_txt]
    for related_node in related_nodes:
        full_context.append(related_node["id"] + ': ' + related_node["text"])
    return full_context

# Answeer the question
def get_answer(question, context):
    query = QA_PROMPT_TEMPLATE.format(question=question, context=context)
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        temperature=0.8,
        model=MODEL_NAME,
    )
    return chat_completion.choices[0].message.content
full_context = build_context(most_relevant_node, most_relevant_txt)
query_answer = get_answer(question, full_context)
print(query_answer)
