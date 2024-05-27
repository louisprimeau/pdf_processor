

import os
import subprocess
import pymupdf
template = [r"\documentclass[10pt]{article}", r"\usepackage{amssymb}", r"\usepackage{amsmath}", r"\usepackage{xcolor}", r"\usepackage[utf8]{inputenc}", r"\usepackage{geometry}", r"\geometry{papersize={80mm,200mm},", r"left=5mm,", r"right=5mm,", r"top=0mm,", r"bottom=0mm}", r"\synctex=1", r"\begin{document}", r"\end{document}"]

text = r"Superconductivity distance blah blah decreases gradually as its temperature is lowered, even down to near absolute zero, abcdefasdfasdf a b c d e f g superconductor a b c d e".split(" "), r"even down to near absolute zero We introduce a method to predict arbitrary, non-parametric probability distributions over the abcd hey there blah blah rotation manifold. This is particularly useful for pose estimation of symmetric and nearly symmetric objects, since the".split(" ")
expression = r"$\textrm{H}_5\textrm{C}_3\textrm{L}_{20}\textrm{Co}_{1-x}\textrm{Be}_{x}\textcolor{white}{.}$"
to_latex = text[0] + [expression] + text[1]

dir_name = '/Users/louisprimeau/Research/LLMS/test_folder'
tex_file_name = 'programmatic'

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
doc = label_char(doc, len(text[0]) + len(template))
#doc = label_char(doc, len(text[0]) + len(template) + 1)
#doc = label_char(doc, len(text[0]) + len(template) - 1)


doc.save('blah.pdf')
