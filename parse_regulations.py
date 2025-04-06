import os
import numpy as np
import json
from bs4 import BeautifulSoup

file_name = 'data/un regulation no 157 – uniform provisions concerning-L_2021082EN.01007501.doc.html.xhtml'
with open(file_name) as fp:
    soup = BeautifulSoup(fp, 'html.parser')

divs = soup.find('body').findChildren('table')
txts = []
req_ids = []
txts_consolidated = []

for div in divs:
    div_txt = div.text.strip()
    subtexts = [s for s in div_txt.splitlines() if s]
    div_txt_after = os.linesep.join(subtexts[1:])
    txt_exists = any(div_txt_after in txt for txt in txts)
    if txt_exists:
        continue
    txts.append(div_txt_after)
    req_id = subtexts[0]
    if not req_id.endswith('.'):
        req_id += '.'
    req_ids.append(req_id)
uniques, indices = np.unique(req_ids, return_index=True)
indices = sorted(indices)
req_ids_array = np.array(req_ids)

for unique_id in indices:
    redundant_ids = np.where(req_ids_array == req_ids[unique_id])[0].tolist()
    if len(redundant_ids) == 1:
        regulation = req_ids[unique_id] + ': ' + txts[unique_id]
        txts_consolidated.append(regulation)
    else:
        regulation = req_ids[unique_id] + ': ' + ' '.join([txts[i] for i in redundant_ids])
        txts_consolidated.append(regulation)

with open('data/un_regulations_157.json', 'w', encoding='utf-8') as f:
    json.dump(txts_consolidated, f, ensure_ascii=False, indent=4)

