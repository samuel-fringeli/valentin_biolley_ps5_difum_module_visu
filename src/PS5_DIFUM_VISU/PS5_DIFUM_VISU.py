"""
PS5_DIFUM_VISU

Module of Visualisation of embedding vectors

Autor : Biolley
Email : valentin.biolley@edu.hefr.ch
Date : 15.12.23
"""
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from transformers import AutoImageProcessor,PreTrainedTokenizerBase


class MyTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
        else:
            raise ValueError("The 'tokenizer' argument must be an instance of PreTrainedTokenizerBase.")

    def tokenize_labels(self, labels):
        """Tokenize the given labels

        Args:
            labels(list[str]): The labels to tokenize
            tokenizer(object): Model tokenizer

        Returns:
            list: The list of tokens
        """
        resArray = []
        for label in labels:
            tokens = self.tokenizer(label, return_tensors="pt")
            resArray.append(self.tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze().tolist()))
        return resArray


class MyModel_text:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    def get_txt_embedding(self, labels):
        """Computes the embeddings for the given labels

    Args:
        labels(list[str]): The labels to encode
        model(object): Text encoder model
        tokenizer(object): Model tokenizer

    Returns:
        tensor: The tensor of encoded labels
        """
        tokens = self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state
        return embeddings


class MyModel_img:
    def __init__(self, model: PreTrainedModel, image_processor: AutoImageProcessor.from_pretrained):
        self.model = model
        self.image_processor = image_processor

    def get_img_embedding(self, urls):
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

            inputs = self.image_processor(image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings_img.append(embedding)

        tensor_embeddings = torch.stack(embeddings_img)
        return tensor_embeddings


def compute_cosine_similarities(embeddings_1, ref_label_idx=0, embeddings_2=None):
    """
    Compute similarities between a reference label and a list of labels using embeddings (txt or images).

    Parameters:
        embeddings_1 (tensor): A tensor for which you want to calculate similarities
        ref_label_idx (int, optional): The index of the reference label for which to calculate similarities. Default is 0
        embeddings_2 (tensor,optional): The tensor who contains the ref_label_idx (if not here, use ref_label of Labels1 )

    Returns:
        Raise ValueError: If tensors haven't the same size
        similarities (numpy.ndarray): An array of similarities corresponding to the labels.
    """
    # Create pythorch cosine similarity instance, use for compute the cosine similarity
    cos = torch.nn.CosineSimilarity(dim=-1)
    # Check if optional parameter embeddings_2 is given
    if embeddings_2 is None:
        # If not compute cosine similarity between embedding_1 and embedding_1
        embeddings_2 = embeddings_1
    # If tensors are not the same size, throw an exception
    if embeddings_1.size(-1) != embeddings_2.size(-1):
        raise ValueError("Size of embeddings 1 (", embeddings_1.shape, ") and size of embedding2(", embeddings_2.shape,
                         ") is not equals")
    reference_vector = embeddings_1[ref_label_idx][-1]

    similarities = []
    # Compute cosine similarities between each embedding vector and the ref_label_idx embedding vector
    if len(embeddings_1.shape) > 2 and len(embeddings_2.shape) > 2:
        for i in range(embeddings_2.size(0)):
            current_vector = embeddings_2[i][-1]
            similarity = cos(reference_vector, current_vector)
            similarities.append(similarity.item())
    else:
        similarities = np.array(
            [cos(embeddings_2[ref_label_idx], embeddings_1[i]).detach().numpy() for i in range(len(embeddings_1))])

    return similarities


def get_TSNE(embeddings, component=2):
    """
    Compute the TSNE of a tensor, reduce the dimension to n_component

    Parameters:
        embeddings (tensor): Tensor who represent the labels

    Returns:
        text_embeddings_TSNE (numpy.ndarray nD): An array nD who represent the tensor.
    """
    # Create sklearn TSNE instance
    if embeddings.size(0) <= 3:
        raise ValueError("Size of embeddings  (", embeddings.size(0), ") is smaller than 3")
    tsne = TSNE(random_state=1, n_components=component, metric="cosine", perplexity=embeddings.size(0) / 2)
    if len(embeddings.shape) > 2:
        total = []
        for i in range(embeddings.size(0)):
            current_vector = embeddings[i][-1].detach().numpy()
            total.append(current_vector)
        total = np.array(total)
        # Apply TSNE
        embeddings = tsne.fit_transform(total)

    else:
        embeddings = embeddings.detach().numpy()
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
    # Create sklearn PCA instance
    pca = PCA(n_components=component)
    if len(embeddings.shape) > 2:
        total = []
        for i in range(embeddings.size(0)):
            current_vector = embeddings[i][-1].detach().numpy()
            total.append(current_vector)
        total = np.array(total)
        embeddings = total
        # Apply PCA
        embeddings = pca.fit_transform(embeddings)
    else:
        embeddings = embeddings.detach().numpy()
        embeddings = pca.fit_transform(embeddings)

    return embeddings


def read_file_label(url_file):
    """Read label in a file

    Args:
        url_file(list[str]): The url of file containing the labels

    Returns:
        lines(list[str]): The list of labels
    """
    # Open file and return each line in an array
    with open(url_file, 'r') as f:
        lines = f.read().splitlines()
        return lines


def read_dir_image(url_dir):
    """Read images present in directory and build urls to access all images

    Args:
        url_dir(list[str]): The url of directory containing the images

    Returns:
        res(list[str]): The list of urls images contain in the directory
    """
    res = []
    # Enumerate all file present in the directory
    files = os.listdir(url_dir)
    # For each file, build url to access all images
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
        path(str): Location where to load the tensor
        label(str): Load only tensor that match with label

    Returns:
        tensor(tensor): Tensor loaded
        label(str): Label of loaded tensor
    """
    # Enumerate all file present in the directory
    files = os.listdir(path)
    labels = []
    tensors = []
    # for each file get only file that match the label
    for file in files:
        file_name = os.path.splitext(os.path.basename(path + file))[0]
        label_split = file_name.split('_', 1)
        if label_split[1] == label:
            labels.append(label_split[1])
            tensors.append(torch.load(path + "/" + file))

    return tensors, labels


def create_data_set_for_vis(embeddings, input_name, label=0, ref_label_idx=0):
    """Computes the dataframe to help the visualisation with plotly, use the return of this function as input of the fonction visualise_embedding

    Args:
        embeddings(list[tensor]): list of tensors to display
        input_name(list[string): List of name of embeddings use for display name of points (same size of embeddings)
        label(list[string],optional): Labels use for display the color of points (same size of embeddings)
        ref_label_idx (int, optional): The index of the reference label for which to calculate similarities. Default is 0
    Returns:
        Raise ValueError: If embeddings,input_name and label haven't the same size
        dataframe: The dataFrame of all data we need to create graph
    """
    # Check size of embeddinds,input_name,label, if not equals raise an error
    if len(embeddings) != len(input_name):
        raise ValueError("Size of embeddings (", len(embeddings), ") and size of prompt(", len(input_name),
                         ") is not equals")
    if label != 0:
        if len(embeddings) != len(label):
            raise ValueError("Size of embeddings (", len(embeddings), ") and size of label(", len(label),
                             ") is not equals")
    # Create empty dataFrame for result
    data_to_vis = pd.DataFrame()
    i = 0
    # For each embedding compute all necessary data for visualisation
    for embedding in embeddings:
        data_to_vis_temp = pd.DataFrame()
        # compute cosine similarity
        similarity = compute_cosine_similarities(embeddings[ref_label_idx], ref_label_idx, embedding)
        # compute TSNE
        txt_embedding_tsne = get_TSNE(embedding)
        txt_embedding_tsne_3_componnent = get_TSNE(embedding, 3)
        # add all this data on the dataFrame
        data_to_vis_temp["embedding_X"] = txt_embedding_tsne[:, 0]
        data_to_vis_temp["embedding_Y"] = txt_embedding_tsne[:, 1]
        data_to_vis_temp["embedding_X_3D"] = txt_embedding_tsne_3_componnent[:, 0]
        data_to_vis_temp["embedding_y_3D"] = txt_embedding_tsne_3_componnent[:, 1]
        data_to_vis_temp["embedding_z_3D"] = txt_embedding_tsne_3_componnent[:, 2]
        data_to_vis_temp["prompt"] = input_name[i]
        data_to_vis_temp["similarity"] = similarity
        # if no label is given
        if label != 0:
            data_to_vis_temp["label"] = label[i]
        # concate this dataFrame to dataframe for result
        data_to_vis = pd.concat([data_to_vis, data_to_vis_temp], ignore_index=True)
        i += 1

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
    # check if label exist in the data
    if 'label' in data:
        # display color according to label
        fig = px.scatter(data, x='embedding_X', y='embedding_Y', color='label', hover_name="prompt")
    else:
        # don't display color
        fig = px.scatter(data, x='embedding_X', y='embedding_Y', hover_name="prompt")

    max = data['similarity'].nlargest(2).index[-1]
    min = data['similarity'].idxmin()
    highlight = ref_label_idx
    # get ref_label_idx sample
    x_highlight = data.loc[highlight, 'embedding_X']
    y_highlight = data.loc[highlight, 'embedding_Y']
    z_highlight = data.loc[highlight, 'similarity']

    x_highlight_min = data.loc[min, 'embedding_X']
    y_highlight_min = data.loc[min, 'embedding_Y']
    z_highlight_min = data.loc[min, 'similarity']

    x_highlight_max = data.loc[max, 'embedding_X']
    y_highlight_max = data.loc[max, 'embedding_Y']
    z_highlight_max = data.loc[max, 'similarity']

    x_highlight_3D_min = data.loc[min, 'embedding_X_3D']
    y_highlight_3D_min = data.loc[min, 'embedding_y_3D']
    z_highlight_3D_min = data.loc[min, 'embedding_z_3D']

    x_highlight_3D = data.loc[highlight, 'embedding_X_3D']
    y_highlight_3D = data.loc[highlight, 'embedding_y_3D']
    z_highlight_3D = data.loc[highlight, 'embedding_z_3D']

    x_highlight_3D_max = data.loc[max, 'embedding_X_3D']
    y_highlight_3D_max = data.loc[max, 'embedding_y_3D']
    z_highlight_3D_max = data.loc[max, 'embedding_z_3D']

    # Display the base sample in green
    fig.add_trace(go.Scatter(x=[x_highlight], y=[y_highlight], mode='markers', marker=dict(size=10, color='green'),
                             name="input de base: " + data["prompt"][ref_label_idx], visible="legendonly"))
    # Display the min cosine similarity sample in green
    fig.add_trace(
        go.Scatter(x=[x_highlight_min], y=[y_highlight_min], mode='markers', marker=dict(size=10, color='orange'),
                   name="Similarity min: " + data["prompt"][min], visible="legendonly"))
    # Display the max cosine similarity sample in green
    fig.add_trace(
        go.Scatter(x=[x_highlight_max], y=[y_highlight_max], mode='markers', marker=dict(size=10, color='yellow'),
                   name="Similarity max: " + data["prompt"][max], visible="legendonly"))

    # Add title on the graphe and name axes
    fig.update_layout(title='Representation vecteurs embedding 2D avec TSNE 2 composantes',
                      xaxis=dict(title='embedding_X'), yaxis=dict(title='embedding_Y'))

    fig.show()
    # check if label exist in the data
    if 'label' in data:
        # display color according to label
        fig = px.scatter_3d(data, x='embedding_X', y='embedding_Y', z='similarity', color='label', hover_name="prompt")
    else:
        # don't display color
        fig = px.scatter_3d(data, x='embedding_X', y='embedding_Y', z='similarity', hover_name="prompt")

    # Display this sample in green
    fig.add_trace(go.Scatter3d(x=[x_highlight], y=[y_highlight], z=[z_highlight], mode='markers',
                               marker=dict(size=10, color='green'),
                               name="input de base: " + data["prompt"][ref_label_idx], visible="legendonly"))

    fig.add_trace(go.Scatter3d(x=[x_highlight_min], y=[y_highlight_min], z=[z_highlight_min], mode='markers',
                               marker=dict(size=10, color='orange'), name="Similarity min: " + data["prompt"][min],
                               visible="legendonly"))

    fig.add_trace(go.Scatter3d(x=[x_highlight_max], y=[y_highlight_max], z=[z_highlight_max], mode='markers',
                               marker=dict(size=10, color='yellow'), name="Similarity max: " + data["prompt"][max],
                               visible="legendonly"))
    # Add title on the graphe and name axes
    fig.update_layout(
        title='Representation vecteurs embedding 3D avec TSNE 2 composante et avec la similaritié sur l\'axe z',
        scene=dict(xaxis=dict(title='embedding_X'), yaxis=dict(title='embedding_Y'), zaxis=dict(title='Similarity')))
    fig.show()
    # check if label exist in the data
    if 'label' in data:
        # display color according to label
        fig = px.scatter_3d(data, x='embedding_X_3D', y='embedding_y_3D', z='embedding_z_3D', color='label',
                            hover_name="prompt")
    else:
        # don't display color
        fig = px.scatter_3d(data, x='embedding_X_3D', y='embedding_y_3D', z='embedding_z_3D', hover_name="prompt")

    fig.add_trace(go.Scatter3d(x=[x_highlight_3D], y=[y_highlight_3D], z=[z_highlight_3D], mode='markers',
                               marker=dict(size=10, color='green'),
                               name="input de base: " + data["prompt"][ref_label_idx], visible="legendonly"))

    fig.add_trace(go.Scatter3d(x=[x_highlight_3D_min], y=[y_highlight_3D_min], z=[z_highlight_3D_min], mode='markers',
                               marker=dict(size=10, color='orange'), name="Similarity min: " + data["prompt"][min],
                               visible="legendonly"))

    fig.add_trace(go.Scatter3d(x=[x_highlight_3D_max], y=[y_highlight_3D_max], z=[z_highlight_3D_max], mode='markers',
                               marker=dict(size=10, color='yellow'), name="Similarity max: " + data["prompt"][max],
                               visible="legendonly"))

    fig.update_layout(title='Representation vecteurs embedding 3D avec TSNE 3 composantes',
                      scene=dict(xaxis=dict(title='embedding_X'), yaxis=dict(title='embedding_Y'),
                                 zaxis=dict(title='embedding_Z')))
    fig.show()
