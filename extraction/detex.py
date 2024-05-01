import os

paper_source_directory = '/home/louis/research/pdf_processor/processed_data/superconductivity_processed/'

for directory in os.listdir(paper_source_directory):
    object_path = os.path.join(paper_source_directory, directory)
    if os.path.isdir(object_path) and not directory.startswith("."):
        paper_textfile = os.path.join(paper_source_directory, directory, 'text.txt')
        out_textfile = os.path.join(paper_source_directory, directory, 'flat.txt')
        os.system('detex {} > {}'.format(paper_textfile, out_textfile))