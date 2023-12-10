# from pdfminer.high_level import extract_text
import pprint
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer


is_syllabus = False
syllabus = {
    "modules": 0
}

texts = []  # extract_text('sample.pdf')
syllabus_list = []

for page_layout in extract_pages("sample.pdf"):
    for index, element in enumerate(page_layout):
        if isinstance(element, LTTextContainer):
            text = element.get_text()
            texts.append(element.get_text())
            print("\n")
            if text.split('\n')[0].strip().lower().startswith('text book') and is_syllabus is True:
                is_syllabus = False
            if is_syllabus:
                syllabus_list.append(text.strip())
            if text.replace("\n", "").strip().lower() == 'syllabus' and is_syllabus is False:
                is_syllabus = True


# texts = texts.replace('\n\n', '\n')
# texts  = texts.replace(' ', '')
# textArray = texts.split('\n')
texts = list(map(str.strip, texts))
texts = list(filter(None, texts))

syllabus['subject_code'] = texts[0]
syllabus['subject_name'] = texts[1].replace('\n', '')

syllabus_start = [syllabus_list.index(text) for text in syllabus_list if text.lower().startswith('module')][0]
syllabus_list_temp = syllabus_list


syllabus_list = []
for text in syllabus_list_temp:
    newline_index = text.find('\n')
    if text.lower().startswith('module') and newline_index > 0:
        syllabus_list.extend([text[:newline_index], text[newline_index:]])
    else:
        syllabus_list.append(text)

syllabus_list = list(filter(None, syllabus_list[syllabus_start::]))


def roman_to_int(s):
    rom_val = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


curr_module = 1
module_content = []


for module in syllabus_list:
    if module.strip().lower().startswith('module'):
        module_index = module.replace('\n', '').replace('-', '').replace('–', '').split()[1]
        curr_module = int(module_index) if module_index.isdigit() else roman_to_int(module_index)
        syllabus['modules'] = curr_module
        if curr_module > 1:
            module_content = list(map(str.strip, module_content))
            syllabus['module_' + str(curr_module-1)] = list(filter(None, module_content))
        module_content = []
    else:
        contents = module.split('.')
        contents = list(filter(None, contents))
        for content in contents:
            colonIndex = content.find(":")
            if colonIndex < 0:
                content = content.replace('\n', '')
                content = content.strip()
                module_content.extend(content.split(','))
            else:
                module_content.append(content.strip())

pprint.pprint(syllabus)
