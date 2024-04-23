
import re
from unidecode import unidecode

# remove whitespace inside math mode expressions
def format_math_whitespace(string):
    substrings = string.split('$')
    start = 1 if string[0] != '$' else 0
    for i in range(start, len(substrings), 2):
        substrings[i] = substrings[i].replace(' ', '')
    return '$'.join(substrings)

# replace hyphen spaces outside of math mode
def replace_hyphen_spaces(string):
    substrings = string.split('$')
    start = 1 if string[0] == '$' else 0
    for i in range(start, len(substrings), 2):
        substrings[i] = re.sub('- (\s*)', '', substrings[i])
    return '$'.join(substrings)

# replace lookalike unicode to ascii
def replace_common_unicode(string):
    return unidecode(string, errors='ignore')


def filter_paragraphs_keywords(text_list, keywords):
    out = []
    for paragraph in text_list:
        if any(keyword in paragraph for keyword in keywords):
            out.append(paragraph)
    return out
