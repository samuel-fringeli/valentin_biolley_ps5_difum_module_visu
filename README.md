# PS5_DIFUM_VISU

> Original author of the project: Biolley Valentin

Ce guide a pour but de démontrer l'utilisation du module de visualisation PS5_DIFUM_VISU qui se trouve sur le repository git suivant : https://gitlab.forge.hefr.ch/valentin.biolley/ps5_difum_module_visu


### Structure:
Ce module de visualisation regroupe les résultats des travaux faits dans le notebook suivant: https://gitlab.forge.hefr.ch/valentin.biolley/detecteur_incendie_ou_de_fumee_utilisant_les_foundation_models/-/blob/main/src/Grounding_dino_simple_NB/PS5_DFUM_VISU.ipynb?ref_type=heads

Différents labels utilisés pour les tests se trouvent dans le dossier "file"
ainsi que des images de tests se trouvent dans le dossier "images_fire_not_fire"

Ensuite, ce module met à disposition des outils permettant de manipuler les vecteurs d'embeddings. La documentation des entrées sorties de chacune de ces fonctions se trouvent dans le notebook.

Voici quelques méthodes utiles pour la générer les données nécessaires à la visualisation :

- get_txt_embedding
- read_file_label
- read_dir_image
- get_img_embedding_swin
- create_data_set_for_vis
- visualise_embedding

## Getting started
### Installation du module
Pour pouvoir installer le module de visualisation, il faut le charger depuis le repository git grâce à la commande suivante :
```bash
pip install git+https://gitlab.forge.hefr.ch/valentin.biolley/ps5_difum_module_visu
```
### Import du module
Voici comment importer le module pour pouvoir l'utiliser.
```python
from PS5_DIFUM_VISU import PS5_DIFUM_VISU as PS5
```

### Exemple d'utilisation
Voici un exemple d'utilisation du module de visualisation. Pour que le module soit le plus modulable possible, le module utilise des classes génériques permettant de prendre en entrée n'import quel modèle permettant la génération des embeddings de text ou d'image, il existe une classe générique pour le text et une pour les images, respectivement MyModel_text et MyModel_img. 

Il y a 3 méthodes abstraites à implémenter :
* get_img_embedding
* get_txt_embedding
* tokenize_labels

### Exemple d'implémentation des méthodes abstraites
```python
from PS5_DIFUM_VISU import PS5_DIFUM_VISU as PS5
class CustomTokenizer(PS5.MyTokenizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
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
    
class CustomModel_text(PS5.MyModel_text):
    def __init__(self, model, tokenizer):
        super().__init__(model,tokenizer)

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
        
class CustomModel_img(PS5.MyModel_img):
    def __init__(self,model, image_processor):
        super().__init__(model, image_processor)
        
    def get_img_embedding(self,urls):
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
                image=Image.open(new_image_path)
            else:
                image = Image.open(image_path)
    
            inputs = self.image_processor(image, return_tensors="pt")
        
    
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings_img.append(embedding)
    
        tensor_embeddings = torch.stack(embeddings_img)
        return tensor_embeddings
```
### Exemple d'utilisation du module de visualisation
```python
from PS5_DIFUM_VISU import PS5_DIFUM_VISU as PS5
# read file to get labels
LABELS_FIRE=PS5.read_file_label("./File/label_fire.txt")
LABELS_FIRE_OPPOSITE=PS5.read_file_label("./File/label_fire_opposite.txt")
# create classe with personal model and tokenizer
model_text=CustomModel_text(model,tokenizer)
# compute embeddings vector of labels
embeddings_txt=model_text.get_txt_embedding(LABELS_FIRE)
embeddings_txt_opposite=model_text.get_txt_embedding(LABELS_FIRE_OPPOSITE)
# name of point
prompt=[LABELS_FIRE,LABELS_FIRE_OPPOSITE]
# color of point
labels=["Fire","Not Fire"]
all_embeddings=[embeddings_txt,embeddings_txt_opposite]
# create dataset to simplify generation of graphe
data_to_visu = PS5.create_data_set_for_vis(all_embeddings,prompt,labels)
# generate graphe with dataset
PS5.visualise_embedding(data_to_visu)
```