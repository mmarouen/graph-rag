import json

with open('data/un_regulations_simple.json') as f:
    regulations = json.load(f)

keys = [regulation["key"] for regulation in regulations]
text_data = [regulation["short_title"] + ' ' + regulation["long_title"] for regulation in regulations]

# print(regulations)
input_query = "wheels"

def retrieve_query(input_query):
    is_in_key = input_query in keys
    result = []
    if is_in_key:
        result.append(input_query)
    for index, txt in enumerate(text_data):
        if input_query in txt:
            result.append(keys[index])
    return result

query_result = retrieve_query(input_query)
print(query_result)