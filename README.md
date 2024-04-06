This repository is some scripts to process physics papers into text files.

The workflow is as follows:

1. Go into main.py and change the source directory to the desired directory.
2. Run main.py with the dependencies in requirements.yml (there may be some extra not strictly necessary packages)
3. The script will walk through all subdirectories of the source directory looking for pdfs.
4. It will run paragraph and figure recognition using the detectron model.
5. It will run text and latex recognition on the extracted paragraphs using the Pix2Text package.
6. The script heuristically decides which paragraphs are captions. It does this by finding the paragraph with the minimum distance between the bottom center of a given figure and the centroid of a paragraph's bounding box.
7. The package runs a gaussian mixture model on the paragraph coordinates to decide whether they belong to the same column an N column layout.  
8. It will combine all non-caption paragraphs and write them to a target directory, which will be named the same thing as the source pdf except without the .pdf extension, inside the output/ directory.
9. It will extract all the figures and write them as X.png into the target directory.
10. It will extract all caption paragraphs and write them to a separate text file named caption.txt, in the same order as the figures.
11. Errors are caught and those pdfs are skipped. The offending pdf's names are written to failed.txt.
