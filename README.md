# Civilization

## Modèle

| Thème   |  Description  | Paramètres | Equation
| -------- | ------- | -------| ---- |
| Géographie  | La carte contient différents sous-espaces géographiques : plaines, montagnes, thalassographie. La carte est discrétisée sous forme de cases.   | ``` movement_speed ``` <br>  ``` fertility ``` | |
| Démographie  | Chaque case a une capacité démographique maximale, qui dépend du niveau technologique.     | ``` population ```  <br>  ``` technological_level ```| |
| Culture    | Chaque unité de population a une “culture” qui évolue dans un espace multidimensionnel. Cette culture accompagnerait les migrants vers les cases voisines. La diffusion culturelle est influencée par le “prestige culturel”, qui dépend du niveau technologique et de l’avancement politique.  | ``` culture_score ```  <br>  ``` political_phase ``` | |
| Progrès    | À chaque unité de temps, la probabilité qu’une case passe au niveau technologique ou politique suivant est calculée en fonction de la densité de population, de l’avancement politique et de la surface culturelle contiguë.   | | |
| Homogénéisation    | Modèle de diffusion/segmentation a élaborer.   | | |
| Migration    | La croissance démographique naturelle peut déborder sur les cases avoisinantes, selon un mécanisme de diffusion. Les populations sont attirées par les zones avec le meilleur prestige culturel   | | |