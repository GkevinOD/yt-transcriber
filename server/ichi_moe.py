import urllib
from bs4 import BeautifulSoup

def parse_html(soup: BeautifulSoup):
    rows = soup.find_all('div', {'class': 'row gloss-row'})
    glosses = []
    for row in rows:
        glosses.extend(row.find_all('div', {'class': 'gloss'}))

    result = []
    for entry in glosses:
        current = {'text': '', 'reading': '', 'definitions': []}
        dt = entry.find('dt')
        dd = entry.find('dd')

        def process_text(in_text):
            # reading is in between of 【 and 】
            out_reading = in_text[in_text.find('【') + 1:in_text.find('】')] if in_text.find('【') != -1 else ''

            # text is before '【' and after '.' if they exist
            out_text = in_text[:in_text.find('【')] if in_text.find('【') != -1 else in_text
            # right of '.' if it exists
            out_text = out_text[out_text.find('.') + 1:] if out_text.find('.') != -1 else out_text
            out_text = out_text.strip()

            # reading is text if it is empty
            out_reading = out_text if out_reading == '' else out_reading
            return out_text, out_reading

        current['text'], current['reading'] = process_text(dt.text)
        current['definitions'] = [x.strip().replace('] note ', '] ') for x in dd.text.split('  ') if x.strip() != '']
        if len(current['definitions']) > 0 and current['definitions'][0] == 'Compound word:':
            # combine 0 and 1
            current['definitions'][0] = current['definitions'][0] + ' ' + current['definitions'][1]
            current['definitions'].pop(1)

        result.append(current)
    return result

def get_ichi_moe(text):
    url = 'https://ichi.moe/cl/qr/?q=' + urllib.parse.quote(text)
    response = urllib.request.urlopen(url)
    html = response.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')

    dict_list = parse_html(soup)
    if len(dict_list) == 0:
        return {'error': 'No results found.'}

    text[0:text.index(dict_list[0]['text'])]
    dict_list.insert(0, {'text': text[0:text.index(dict_list[0]['text'])], 'reading': '', 'definitions': []})

    text_index = 0
    for i in range(0, len(dict_list) - 1):
        index = text.index(dict_list[i]['text'], text_index) + len(dict_list[i]['text'])
        text_index += len(dict_list[i]['text'])
        next_index = text.index(dict_list[i + 1]['text'], index)
        extra = text[index:next_index]
        if extra != '':
            dict_list.insert(i + 1, {'text': extra, 'reading': '', 'definitions': []})

    text[text.index(dict_list[-1]['text'], text_index) + len(dict_list[-1]['text']):]
    dict_list.append({'text': text[text.index(dict_list[-1]['text'], text_index) + len(dict_list[-1]['text']):], 'reading': '', 'definitions': []})

    return dict_list