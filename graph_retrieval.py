import os
from langchain_groq import ChatGroq
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddingss

import json

FILENAME = 'data/un_regulations_157_graph.json'
USERNAME = 'neo4j'
PASSWORD = 'ADM16600'
URI = 'neo4j://localhost:7687'
MODEL_NAME = 'llama3-70b-8192'
OPENAI_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcHAiLCJleHAiOjE3OTk5OTk5OTksInN1YiI6MTc1NTM2MywiYXVkIjoiV0VCIiwiaWF0IjoxNjk0MDc2ODUxfQ.4ySh5zSCHGDMkitFWvWMPXdzwYdMY-0Dc33vpMjCgZA'
CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to 
query a graph database.
Context:
The graph database contains a list of automotive regulation compliance text, each node represents a requirement or a definition.
The node has a text attribute which contains the actual regulation.
The relationships represent either a hierarchical relationship (node 3.2.1 is child_of 3.2) or a mention relation (node 4.2.5.1 mentions regulation 2.1.0)
Your task is to perform for each regulation the following actions in order:
- identify the id of regulation (located at the beginning of the regulation such as 1.2.3 or 2.6 ...)
- the id of the regulation must be reported in double quotes ("1.2.3", "2.6"...)
- identify any mention of other regulations within the text. example "regard to the items mentioned in paragraph 2.1.1"
Instructions:
- Identify the most relevant node to the query, pay attention to the dependance relationships that may hide extra dependencies w.r.t. the query
- Whenever relevant, always return the requirement id and text so that the requirement could be identified by its id and content.
- Use only the provided relationship types and properties in the  schema. Do not use any other relationship types or properties that  are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than 
for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher 
statements for particular questions:

# Which requirements relate to mechanical safety?
MATCH (n1:Node)-[:mentions]->(n2:Node)
WHERE n1.text CONTAINS 'mechanical safety'
RETURN n1.id, n1.text, n2.id, n2.text
The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

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

# driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
graph = Neo4jGraph(url=URI, username=USERNAME, password=PASSWORD)

'''
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

create_node_query = """
MERGE (n:Node {id: $id})
SET n.text = $text
RETURN n
"""
create_relation_query = """
MATCH (a:Node {id: $source}), (b:Node {id: $destination})
MERGE (a)-[r:`%s`]->(b)
RETURN r
"""
for node in data:
    graph.query(create_node_query, {"id": node["id"], "text": node["text"]})

# Create Relationships
for node in data:
    for relation in node["relations"]:
        source, relationship, destination = relation
        dynamic_query = create_relation_query % relationship
        graph.query(dynamic_query, {
            "source": source,
            "destination": destination
        })

graph.refresh_schema()
print(graph.schema)
'''
graph.refresh_schema()
print(graph.schema)

llm_client = ChatGroq(temperature=0, groq_api_key=os.environ.get("API_KEY"), model_name=MODEL_NAME)
#llm_client = ChatOpenAI(temperature=0, api_key=OPENAI_KEY)

# graph_store = Neo4jGraphStore(graph)
chain = GraphCypherQAChain.from_llm(
    llm_client,
    graph=graph,
    allow_dangerous_requests=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    qa_prompt=QA_GENERATION_PROMPT,
    verbose=True)

question = "List requirments related to transition phase"
prompt = CYPHER_GENERATION_TEMPLATE.format(schema=graph.schema, question=question)
# print(prompt)
reponse = chain.run({"query": question})
print(reponse)
