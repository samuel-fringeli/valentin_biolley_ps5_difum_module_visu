import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os


def tokenizerLabels(labels, tokenizer):
    """Tokenize the given labels

    Args:
        labels(list[str]): The labels to tokenize
        tokenizer(object): Model tokenizer

    Returns:
        list: The list of tokens
    """
    resArray = []
    for label in labels:
        # Tokenization
        tokens = tokenizer(label, return_tensors="pt")
        resArray.append(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist()))
    return resArray


def get_txt_embedding(labels, model, tokenizer):
    """Computes the embeddings for the given labels

    Args:
        labels(list[str]): The labels to encode

    Returns:
        tensor: The tensor of encoded labels
    """
    tokens = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)
    # Obtaining the embeddings
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state

    return embeddings


def compute_cosine_similarities(embeddings_1, ref_label_idx=0, embeddings_2=None):
    """
    Compute similarities between a reference label and a list of labels using text embeddings.

    Parameters:
        embeddings_1 (tensor): A tensor for which you want to calculate similarities
        ref_label_idx (int, optional): The index of the reference label for which to calculate similarities. Default is 0
        embeddings_2 (tensor,optional): The tensor who contains the ref_label_idx (if not here, use ref_label of Labels1 )

    Returns:
        similarities (numpy.ndarray): An array of similarities corresponding to the labels.
    """
    cos = torch.nn.CosineSimilarity(dim=-1)
    if embeddings_2 is None:
        embeddings_2 = embeddings_1
    if embeddings_1.size() != embeddings_2.size():
        raise ValueError("Size of embeddings 1 (", embeddings_1.shape, ") and size of embedding2(", embeddings_2.shape,
                         ") is not equals")
    embeddings_1 = embeddings_1.view(embeddings_1.size(0), -1)
    embeddings_2 = embeddings_2.view(embeddings_2.size(0), -1)

    similarities = np.array(
        [cos(embeddings_2[ref_label_idx], embeddings_1[i]).detach().numpy() for i in range(len(embeddings_1))])

    return similarities


def get_TSNE(embeddings, component=2):
    """
    Compute the TSNE of a tensor, reduce the dimension to n_component

    Parameters:
        embeddings (tensor): Tensor who represent the labels

    Returns:
        text_embeddings_TSNE (numpy.ndarray nD): An array nD who represente the tensor.
    """
    embeddings = embeddings.view(embeddings.size(0), -1)
    embeddings = embeddings.detach().numpy()
    tsne = TSNE(random_state=1, n_components=component, metric="cosine", perplexity=2.0)
    embeddings = tsne.fit_transform(embeddings)
    return embeddings


def get_PCA(embeddings, component=2):
    """
    Compute the PCA of a tensor, reduce the dimension to n_component.

    Parameters:
        embeddings (tensor): Tensor who represent the labels

    Returns:
        text_embeddings_PCA (numpy.ndarray nD): An array nD who represent the tensor.
    """
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    embeddings = embeddings.detach().numpy()
    pca = PCA(n_components=component)
    embeddings = pca.fit_transform(embeddings)
    return embeddings


def get_img_embedding_swin(urls, model, image_processor):
    """Computes the embeddings for the given image locate in urls

    Args:
        urls(list[str]): The urls of images to encode

    Returns:
        tensor: The tensor of encoded images
    """

    embeddings_img = []

    for image_path in urls:
        if image_path.lower().endswith('.png'):
            image = Image.open(image_path)
            image = image.convert('RGB')
            new_image_path = os.path.splitext(image_path)[0] + ".jpg"
            image.save(new_image_path)
            image = Image.open(new_image_path)
        else:
            image = Image.open(image_path)

        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings_img.append(embedding)

    tensor_embeddings = torch.stack(embeddings_img)
    return tensor_embeddings


def read_file_label(url_file):
    """Read label in a file

    Args:
        url_file(list[str]): The url of file containing the labels

    Returns:
        lines(list[str]): The list of labels
    """
    with open(url_file, 'r') as f:
        lines = f.read().splitlines()
        return lines


def read_dir_image(url_dir):
    """Read label in a file

    Args:
        url_dir(list[str]): The url of directory containing the images

    Returns:
        res(list[str]): The list of images
    """
    res = []
    files = os.listdir(url_dir)
    for file in files:
        final_url = url_dir + file
        res.append(final_url)
    return res


def save_tensor(embedding, label, path):
    """Save a tensor on /path_label.pt on disk

    Args:
        embedding(tensor): Tensor to save
        label (str): label of tensor to save
        path (str): location where to save the tensor

    Returns:
        -
    """
    torch.save(embedding, path + "_" + label + ".pt")


def load_tensor(path, label):
    """Load tensor from disk

    Args:
        path(str): location where to load the tensor

    Returns:
        tensor(tensor): Tensor loaded
        label(str): Label of loaded tensor
    """
    files = os.listdir(path)
    labels = []
    tensors = []
    for file in files:
        file_name = os.path.splitext(os.path.basename(path + file))[0]
        label_split = file_name.split('_', 1)
        if label_split[1] == label:
            labels.append(label_split[1])
            tensors.append(torch.load(path + "/" + file))

    return tensors, labels


def create_data_set_for_vis(embeddings, embeddings_2, input_name, label, ref_label_idx=0):
    """Computes the dataframe to help the visualisation with plotly

    Args:
        embeddings(tensor): first tensor to display
        embeddings_2(tensor): Second tensor to display
        input_name(list[string): List of name of embeddings use for display name of points
        label(list[string]): Labels use for display the color of points
        ref_label_idx (int, optional): The index of the reference label for which to calculate similarities. Default is 0


    Returns:
        dataframe: The dataFrame of all data we need to create graph
    """
    similarity = compute_cosine_similarities(embeddings, ref_label_idx)
    similarity2 = compute_cosine_similarities(embeddings, ref_label_idx, embeddings_2)

    data_to_vis1 = pd.DataFrame()
    txt_embedding_tsne = get_TSNE(embeddings)
    txt_embedding_tsne_3_componnent = get_TSNE(embeddings, 3)

    data_to_vis1["embedding_X"] = txt_embedding_tsne[:, 0]
    data_to_vis1["embedding_Y"] = txt_embedding_tsne[:, 1]
    data_to_vis1["embedding_X_3D"] = txt_embedding_tsne_3_componnent[:, 0]
    data_to_vis1["embedding_y_3D"] = txt_embedding_tsne_3_componnent[:, 1]
    data_to_vis1["embedding_z_3D"] = txt_embedding_tsne_3_componnent[:, 2]
    data_to_vis1["prompt"] = input_name[0]
    data_to_vis1["similarity"] = similarity
    data_to_vis1["label"] = label[0]

    data_to_vis2 = pd.DataFrame()
    txt_embedding_tsne = get_TSNE(embeddings_2)
    txt_embedding_tsne_3_componnent = get_TSNE(embeddings_2, 3)

    data_to_vis2["embedding_X"] = txt_embedding_tsne[:, 0]
    data_to_vis2["embedding_Y"] = txt_embedding_tsne[:, 1]
    data_to_vis2["embedding_X_3D"] = txt_embedding_tsne_3_componnent[:, 0]
    data_to_vis2["embedding_y_3D"] = txt_embedding_tsne_3_componnent[:, 1]
    data_to_vis2["embedding_z_3D"] = txt_embedding_tsne_3_componnent[:, 2]
    data_to_vis2["prompt"] = input_name[1]
    data_to_vis2["similarity"] = similarity2
    data_to_vis2["label"] = label[1]

    data_to_vis = pd.concat([data_to_vis1, data_to_vis2], ignore_index=True)
    return data_to_vis


def visualise_embedding(data, ref_label_idx=0):
    """Computes the graphe and display it
        - 2D graph with embeddings and 2 components TSNE
        - 3D graph with embeddings and 2 components TSNE and cosine similarity on axe z
        - 3D graph with embeddings and  components TSNE

    Args:
        data(dataFrame): dataFrame generate with create_data_set_for_vis  to help to display
        ref_label_idx (int, optional): label index for which we are going to highlight . Default is 0
    Returns:
        -
    """
    fig = px.scatter(data, x='embedding_X', y='embedding_Y', color='label', hover_name="prompt")

    highlight = ref_label_idx

    x_highlight = data.loc[highlight, 'embedding_X']
    y_highlight = data.loc[highlight, 'embedding_Y']

    fig.add_trace(go.Scatter(x=[x_highlight], y=[y_highlight], mode='markers', marker=dict(size=10, color='green'),
                             name="input de base: " + data["prompt"][ref_label_idx], visible="legendonly"))

    fig.update_layout(title='Representation vecteurs embedding 2D avec TSNE 2 composantes',
                      xaxis=dict(title='embedding_X'), yaxis=dict(title='embedding_Y'))

    fig.show()

    fig = px.scatter_3d(data, x='embedding_X', y='embedding_Y', z='similarity', color='label', hover_name="prompt")

    highlight = ref_label_idx

    x_highlight = data.loc[highlight, 'embedding_X']
    y_highlight = data.loc[highlight, 'embedding_Y']
    z_highlight = data.loc[highlight, 'similarity']

    fig.add_trace(go.Scatter3d(x=[x_highlight], y=[y_highlight], z=[z_highlight], mode='markers',
                               marker=dict(size=10, color='green'),
                               name="input de base: " + data["prompt"][ref_label_idx], visible="legendonly"))

    fig.update_layout(
        title='Representation vecteurs embedding 3D avec TSNE 2 composante et avec la similaritié sur l\'axe z',
        scene=dict(xaxis=dict(title='embedding_X'), yaxis=dict(title='embedding_Y'), zaxis=dict(title='Similarity')))
    fig.show()

    fig = px.scatter_3d(data, x='embedding_X_3D', y='embedding_y_3D', z='embedding_z_3D', color='label',
                        hover_name="prompt")

    highlight = ref_label_idx

    x_highlight = data.loc[highlight, 'embedding_X_3D']
    y_highlight = data.loc[highlight, 'embedding_y_3D']
    z_highlight = data.loc[highlight, 'embedding_z_3D']

    fig.add_trace(go.Scatter3d(x=[x_highlight], y=[y_highlight], z=[z_highlight], mode='markers',
                               marker=dict(size=10, color='green'),
                               name="input de base: " + data["prompt"][ref_label_idx], visible="legendonly"))

    fig.update_layout(title='Representation vecteurs embedding 3D avec TSNE 3 composantes',
                      scene=dict(xaxis=dict(title='embedding_X'), yaxis=dict(title='embedding_Y'),
                                 zaxis=dict(title='embedding_Z')))
    fig.show()
