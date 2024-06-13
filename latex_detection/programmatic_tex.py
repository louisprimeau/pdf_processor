import os
import subprocess
import pymupdf
from wonderwords import RandomWord
import random
import periodictable

gen = RandomWord()

def generate_pdf(name, to_latex, splits):
    # to-do: add fonts and adjust for overflow
    template = [r"\documentclass[10pt]{article}", r"\usepackage{amssymb}", r"\usepackage{amsmath}", r"\usepackage{xcolor}", r"\usepackage[utf8]{inputenc}", r"\usepackage{geometry}", r"\geometry{papersize={80mm,200mm},", r"left=5mm,", r"right=5mm,", r"top=0mm,", r"bottom=0mm}", r"\synctex=1", r"\begin{document}", r"\end{document}"]

    # output file for data
    dir_name = '/home/ephalen/pdf_processor/data'
    tex_file_name = name

    with open(os.path.join(dir_name, tex_file_name + '.tex'), 'w') as f:
        f.write("\n".join(template[0:-1] + to_latex + template[-1:]))

    os.system('pdflatex --synctex=1 -halt-on-error -output-directory={} {}'.format(dir_name, os.path.join(dir_name, tex_file_name + '.tex')))


    def label_char(doc, index):
        out = os.popen('synctex view -i {}:{}:{} -o {}'.format(index, 0, os.path.join(dir_name, tex_file_name + '.tex'), os.path.join(dir_name, tex_file_name + '.pdf'))).read()
        outputs = out.split("Output:")[2:]
        min_x = 10000
        max_x = -10000
        max_y = -10000
        min_y = 10000
        for i, output in enumerate(outputs):
            if i == 0: print(output)
            print(output.split("\n"))
            rect_dict = {o.split(":")[0]: o.split(":")[1] for o in output.split("\n")[1:-2]}
            rect_dict["x"] = float(rect_dict["x"])
            if rect_dict["x"] < min_x: min_x = rect_dict["x"]
            if rect_dict["x"] > max_x: max_x = rect_dict["x"]
            rect_dict["y"] = float(rect_dict["y"])
            rect_dict["H"] = float(rect_dict["H"])
            rect_dict["W"] = float(rect_dict["W"])

            if rect_dict["y"] - rect_dict["H"] < min_y: min_y = rect_dict["y"] - rect_dict["H"]
            if rect_dict["y"] > max_y: max_y = rect_dict["y"]

            #bb = [rect_dict["x"] , rect_dict["y"] - rect_dict["H"], rect_dict["x"] + rect_dict["W"], rect_dict["y"]]
            #doc[0].draw_rect(bb,  color = (0, i / len(outputs), 0), width = 0.5)

        doc[0].draw_rect([min_x, min_y, max_x, max_y], color=(1, 0, 0), width=0.5)
        return doc
    doc = pymupdf.open(os.path.join(dir_name, tex_file_name + '.pdf'))
    for split in splits:
        doc = label_char(doc, split + len(template))
    #doc = label_char(doc, len(text[0]) + len(template) + 1)
    #doc = label_char(doc, len(text[0]) + len(template) - 1)

    doc.save(dir_name + '/' + tex_file_name +'_full.pdf')
    # to-do: image augmentation can go here to increase robustness

def generate_sentence():
    # length of sentence
    length = random.randrange(10, 30)
    words = gen.random_words(length)
    # add referrences [1]
    for i in range(random.randrange(0, 4)):
        words.insert(random.randrange(0, len(words) - 1), "["+ str(random.randrange(1, 50)) +"]")
    # adds random numbers
    for i in range(random.randrange(0, 3)):
        words.insert(random.randrange(0, len(words) - 1), str(random.randrange(1, 9999)))
    return words

def generate_formulas():
    res = r"$"
    # range of potential lenth of chemical formula
    for j in range(random.randrange(1, 6)):
        # to-do: maybe weight numbers higher than x
        subs = [str(random.randrange(1, 20)), "x"]
        ops = ["-", "+"]
        res += r"\textrm{" + periodictable.elements[random.randrange(1, 118)].symbol + "}_{"
        for i in range(random.choice([0, 1, 3])):
            # alternate starting from a number or var to operation
            if i % 2 != 1:
                if len(subs) != 1:
                    sub = random.randrange(0, len(subs))
                else:
                    sub = 0
                res += subs[sub]
                # avoid reusing "types" like 1-1
                subs.pop(sub)
            else:
                res += ops[random.randrange(0, len(ops))]
        res+= "}"
    res += r"\textcolor{white}{.}$"
    return res

# total number of samples to generate
num_pdfs = 5

for i,j in enumerate(range(num_pdfs)):
    to_latex = generate_sentence()
    splits = []
    for l in range(random.randrange(1, 6)):
        splits.append(len(to_latex))
        expression = generate_formulas()
        to_latex += [expression] + generate_sentence()
    generate_pdf(str(i), to_latex, splits)

