# PS5_DIFUM_VISU

> Original author of the project: Biolley Valentin

Ce guide a pour but de démontrer l'utilisation du module de visualisation PS5_DIFUM_VISU qui se trouve sous la forme d'un notebook.

Veuillez suivre le readme.md d'installation du projet.

### Structure:
Lien vers le notebook: https://gitlab.forge.hefr.ch/valentin.biolley/detecteur_incendie_ou_de_fumee_utilisant_les_foundation_models/-/blob/main/src/Grounding_dino_simple_NB/PS5_DFUM_VISU.ipynb?ref_type=heads 

Le notebook utilise les encodeurs Bert pour le text et Swin pour l'image. 

Différents labels utilisés pour les tests se trouvent dans le dossier "file"
ainsi que des images de tests se trouvent dans le dossier "images_fire_not_fire"

Ensuite ce notebook met à disposition pleins d'outils permettant de manipuler les vecteurs d'embeddings. La documentation des entrées sorties de chaqu'une de ces fonctions se trouvent dans le notebook.

Voici quelques méthodes utiles pour la générer les données nécessaires à la visualisation :

- get_txt_embedding
- read_file_label
- read_dir_image
- get_img_embedding_swin

### Utilisation:

Pour pouvoir visualiser des vecteurs d'embeddings, vous n'avez besoin que des fonctions suivantes :

- **create_data_set_for_vis** qui permet de créer un dataset avec toutes les données nécessaires pour la génération des graphiques. 
- **visualise_embedding** qui permet de générer les graphiques.

1 exemple de visualisation se trouvent à la fin du notebook dans le chapitre Test module visu.
### Installation du module
Pour pouvoir installer le module de visualisation, il faut le charger depuis le git grâce à la commande suivante :
pip install git+https://gitlab.forge.hefr.ch/valentin.biolley/ps5_difum_module_visu

### Structure du module:
Le module donne accès aux mêmes fonctions présentent dans le noteBook, elles sont décrites ci-dessus.
