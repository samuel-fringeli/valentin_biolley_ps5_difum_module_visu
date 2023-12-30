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
Voici un exemple d'utilisation du module de visualisation. Pour que le module soit le plus modulable possible, le module utilise des classes génériques permettant de prendre en entrée n'import quel modèle permettant la génération des embeddings de text ou d'image, il existe une classe générique pour le text et une pour les images, respectivement MyModel_text et MyModel_img
```python
from PS5_DIFUM_VISU import PS5_DIFUM_VISU as PS5
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# read file to get labels
LABELS_FIRE=PS5.read_file_label("./File/label_fire.txt")
LABELS_FIRE_OPPOSITE=PS5.read_file_label("./File/label_fire_opposite.txt")
# create classe with personal model and tokenizer
model_text=PS5.MyModel_text(model,tokenizer)
# compute embeddings vector of labels
embeddings_txt=model_text.get_txt_embedding(LABELS_FIRE)
embeddings_txt_opposite=model_text.get_txt_embedding(LABELS_FIRE_OPPOSITE)
# name of point
prompt=[LABELS_FIRE,LABELS_FIRE_OPPOSITE]
# color of point
labels=["Fire","Not Fire"]
all_embeddings=[embeddings,embeddings_opposite]
# create dataset to simplify generation of graphe
data_to_visu = PS5.create_data_set_for_vis(all_embeddings,prompt,labels)
# generate graphe with dataset
PS5.visualise_embedding(data_to_visu)
```