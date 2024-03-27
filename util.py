import numpy as np
import sklearn.mixture
from PIL import Image
def extract_text_from_image(text_model, paragraph_layout, page_image, image_cleaning_pipeline=[], text_cleaning_pipeline=[]):
    output_strings = []
    for block in paragraph_layout:
        if block.type=='Text':

            x1, y1 = int(block.block.x_1), int(block.block.y_1)
            x2, y2 = int(block.block.x_2), int(block.block.y_2)
            paragraph_image = page_image[y1:y2, x1:x2]

            for func in image_cleaning_pipeline: paragraph_image = func(paragraph_image)
            
            paragraph_text = text_model(Image.fromarray(paragraph_image))
            for func in text_cleaning_pipeline:
                paragraph_text = func(paragraph_text)
            
            output_strings.append(paragraph_text)
    return output_strings

def extract_image_from_image(image_layout, page_image, image_cleaning_pipeline=[]):
    output_images = []
    for block in image_layout:
        x1, y1 = int(block.block.x_1), int(block.block.y_1)
        x2, y2 = int(block.block.x_2), int(block.block.y_2)
        figure_image = page_image[y1:y2, x1:x2]
        for func in image_cleaning_pipeline:
            figure_image = func(figure_image)
        output_images.append(figure_image)
    return output_images


# decide reading order of paragraphs
def reorganize_layout(paragraph_layout):
    try:
        x_coords = np.array([t.block.x_1 for t in paragraph_layout])
        y_coords = np.array([t.block.y_1 for t in paragraph_layout]) / 10
        coords = np.vstack((x_coords, y_coords)).T
        idxs = np.arange(len(paragraph_layout))
        model = sklearn.mixture.BayesianGaussianMixture(n_components=2).fit(coords)
        membership = model.predict(coords).reshape(-1)
        clusters = []
        x_centroids = []
        for m in np.unique(membership):
            x_centroids.append(np.mean(x_coords[membership == m]))
            clusters.append(idxs[membership == m][np.argsort(y_coords[membership == m])])
        clusters = [clusters[i] for i in np.argsort(np.array(x_centroids))]
        reordered_idxs = np.hstack(clusters)
        paragraph_layout = [paragraph_layout[i] for i in reordered_idxs]
        return paragraph_layout
    except:
        return paragraph_layout


# I will say that the caption is the text box with centroid
# closest to the middle point of the bottom edge of the figure.
def associate_captions(paragraph_layout, image_layout):
    text_centroids = [((block.block.x_1 + block.block.x_2)/2, (block.block.y_1 + block.block.y_2)/2) for block in paragraph_layout]
    image_bottom_centers = [((block.block.x_1 + block.block.x_2)/2, block.block.y_2) for block in image_layout]
    # forget vectorization

    caption_idxs = list(range(len(image_bottom_centers)))
    for i, (c_x, c_y) in enumerate(image_bottom_centers):
        min_r = 1e12
        for j, (t_x, t_y) in enumerate(text_centroids):
            r = np.sqrt((t_x - c_x)**2 + (t_y - c_y)**2)
            if r < min_r:
                caption_idxs[i] = j
                min_r = r
    return [paragraph_layout[j] for j in caption_idxs], \
           [paragraph_layout[j] for j in range(len(paragraph_layout)) if j not in caption_idxs]
            
    



# pad paragraph image with white margin, improves inference on edges
def pad_paragraph_image(paragraph_image):
    H, W, C = paragraph_image.shape
    pad_prop = 0.1
    pad_H, pad_W, pad_C = int(pad_prop * H), int(pad_prop * W), 0       
    return np.pad(paragraph_image,
                  ((pad_H, pad_H),(pad_W, pad_W),(pad_C, pad_C)),
                  'constant',
                  constant_values=((255, 255),(255, 255),(255, 255))
                  )
