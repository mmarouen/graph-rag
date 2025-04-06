import os
import ast
import json
from groq import Groq

FILENAME = 'data/un_regulations_157.json'
MODEL_NAME = 'llama3-70b-8192'
BATCH_SIZE = 10

PROMPT_MESSAGE = """Following text contains a list of automotive regulation compliance text, most of the time its either a requirement or a definition.
Your task is to perform for each regulation the following actions in order:
- identify the id of regulation (located at the beginning of the regulation such as 1.2.3 or 2.6 ...)
- the id of the regulation must be reported in double quotes ("1.2.3", "2.6"...)
- identify any mention of other regulations within the text. example "regard to the items mentioned in paragraph 2.1.1"
- identify entity relations, there is only 1 relation to consider:
1. "mentions" relation: "3.2.1.\nA description of the vehicle type with regard to the items mentioned in paragraph 2.1.1," so 3.2.1 mentions 2.1.1
The "mentions" relation is optional and it should only be listed if the regulation clearly mentions another regulation.
The output must solely consist in one or more lists in the format [REGULATION_ID, RELATION, REGULATION_ID] for each regulation.
If no relations are found return an empty list.
The output must consist solely of a json format string where the keys are the regulation id and the value is a list of lists of relationships.
Since the output will be parsed as a dict, no additional text should be added to the output.
Example:
###prompt: [
    '3.2.1.\nA description of the vehicle type with regard to the items mentioned in paragraph 2.1.1, together with a documentation package as required in Annex 4 which gives access to the basic design of the ALKS and the means by which it is linked to other vehicle systems or by which it directly controls output variables. The numbers and/or symbols identifying the vehicle type shall be specified.',
    '2.\nGeneral description of the different risks and measures put in place to mitigate these riskss',
    '5.2.3.3.\nThe activated system shall detect the distance to the next vehicle in front as defined in paragraph 7.1.1 and shall adapt the vehicle speed in order to avoid collision.\nWhile the ALKS vehicle is not at standstill, the system shall adapt the speed to adjust the distance to a vehicle in front in the same lane to be equal or greater than the minimum following distance.'
]
###output: 
{{
    "3.2.1": [
        ["3.2.1", "mentions", "2.1.1"],
        ["3.2.1", "mentions", "Annex 4"]
        ],
    "2.": [],
    "5.2.3.3": [
        ["5.2.3.3", "mentions", "7.1.1"]
    ]
}}
###prompt: {regulations}
###output:"""


destination_file = os.path.splitext(FILENAME)[0] + '_graph.json'
with open(FILENAME) as f:
    regulations = json.load(f)

nodes = []
reg_ids = []
for id, regulation in enumerate(regulations):
    reg_id = regulation.split(':')[0]
    reg_txt = ' '.join(regulation.split(':')[1:])
    parents = reg_id.split('.')
    relations = []
    if len(parents) > 2:
        parent = '.'.join(parents[:(len(parents) - 2)]) + '.'
        relations.append([reg_id, 'child_of', parent])
    nodes.append({'id': reg_id, 'text': reg_txt, 'relations': relations})
    reg_ids.append(reg_id)

llm_client = Groq(api_key=os.environ.get("API_KEY"))
for i in range(20, len(regulations), BATCH_SIZE):
    sample_regulations = regulations[i: i + BATCH_SIZE]
    print(f"i {i} starting regulation {sample_regulations[0]}")
    # sample_regulations = regulations[40:50]
    prompt_msg = PROMPT_MESSAGE.format(regulations=sample_regulations)
    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_msg
            }
        ],
        model=MODEL_NAME,
        # temperature=1
    )
    result = chat_completion.choices[0].message.content
    try:
        result_preprocess = result.split('{')
        result = '{\n' + result_preprocess[1]
        result_preprocess = result.split('}')
        result = result_preprocess[0] + '\n}'
        dict_res = ast.literal_eval(result)
        for reg_id, values in dict_res.items():
            if values and (reg_id in reg_ids):
                reg_id_index = reg_ids.index(reg_id)
                for value in values:
                    nodes[reg_id_index]['relations'].append(value)
    except Exception as e:
        print(f'Failed parsing:\nError {e}\nResult {result}')
        break

with open(destination_file, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
