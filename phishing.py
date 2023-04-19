# Etudiants: Clément Basdevant, Baptiste Garcia 
# Encadrant: Yassine Zniyed

import numpy as np
import cv2
import tensorflow as tf
import streamlit as st


st.markdown("""
    <style>
        body {
            background-color: #f0f0f0;
        }
        .stApp {
            max-width: 800px;
            padding: 10px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# Charger le modèle de classification entraîné
model = tf.keras.models.load_model('nouveau_modele2.h5')

# Définir les labels des classes
labels = ['Black Sea Sprat (Clupeonella)', 'Gilt-Head Bream (Dorade royale)', 'Hourse Mackerel (Chinchard)', "Pas de poisson dÃ©tectÃ©", 'Red Mullet (Rouget)', 'Red Sea Bream (Dorade rose)', 'Sea Bass (Bar)', 'Shrimp (Crevette)', 'Striped Red Mullet (Rouget-barbet de roche)', 'Trout (Truite)']

# Définir une fonction pour prédire la classe d'une image donnée
def predict(image):
    # Prétraiter l'image
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    # Faire la prédiction
    prediction = model.predict(np.array([image]))
    
    # Récupérer la probabilité de chaque classe prédite
    probabilities = tf.nn.softmax(prediction[0])
    
    # Récupérer la classe prédite
    class_idx = np.argmax(probabilities)
    
    return labels[class_idx], probabilities[class_idx]

# Définir la page Streamlit
def app():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    #st.set_page_config(page_title="Classification d'images de poissons", page_icon=":fish:", layout="wide")
    st.title("Classification d'images de poissons")
    st.markdown("#### Etudiants: Clément Basdevant, Baptiste Garcia")
    st.markdown("#### Encadrant: Yassine Zniyed")

    
    # Charger l'image à tester
    uploaded_file = st.file_uploader("Choisissez une image de poisson à classer", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Lire l'image
        image = uploaded_file.read()
        image = np.asarray(bytearray(image), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Afficher l'image
        st.image(image[:, :, ::-1], caption='Image de poisson', use_column_width=True)
        # Faire la prédiction
        label, probability = predict(image)
        # Afficher le label de la classe prédite et sa probabilité
        st.write(f"Classe prédite: {label}")
        st.write(f"Probabilité: {probability * 100:.2f}%")
        # On se fait des recettes de cuisine
        if label == 'Sea Bass (Bar)':  # Vérifier si l'image est une classe de bar
          # Afficher la recette du bar
          st.markdown("<h3 style='text-align: center;'>Bar:</h3>", unsafe_allow_html=True)
          st.write("Le bar est un poisson très apprécié en cuisine, particulièrement en Méditerranée. Aussi connu sous le nom de loup de mer, ce poisson blanc à la chair ferme est souvent apprécié grillé, cuit au four ou même en tartare. Le bar est également une source de protéines de qualité et de nutriments essentiels tels que les acides gras oméga-3 et les vitamines B12 et D.")
          st.write("# Fiche technique sur le bar")
          st.write("Nom : Bar (Dicentrarchus labrax)")
          st.write("Apparence : Poisson blanc avec une peau argentÃ©e et des Ã©cailles sombres. La taille varie de 20 Ã  100 cm.")
          st.write("Habitat : Le bar est prÃ©sent dans les eaux cÃ´tiÃ¨res et les estuaires de l'Atlantique, de la MÃ©diterranÃ©e et de la mer Noire.")
          st.write("RÃ©gime alimentaire : Le bar se nourrit de petits poissons, de crustacÃ©s et de cÃ©phalopodes.")
          st.write("Valeur nutritive : Le bar est riche en protÃ©ines de haute qualitÃ©, en acides gras omÃ©ga-3 et en vitamines B12 et D.")
          st.write("PrÃ©paration culinaire : Le bar est souvent grillÃ© ou cuit au four, mais peut Ã©galement Ãªtre prÃ©parÃ© en tartare ou en ceviche.")
          st.write("SaisonnalitÃ© : Le bar est gÃ©nÃ©ralement disponible toute l'annÃ©e, mais sa saison de pÃªche optimale se situe entre le printemps et l'automne.")
                      
          st.write("# Recette de bar")
          st.write("IngrÃ©dients :")
          st.write("- 1 bar entier vidÃ© et Ã©caillÃ©")
          st.write("- 2 citrons")
          st.write("- 3 gousses d'ail")
          st.write("- 4 branches de thym frais")
          st.write("- 4 cuillÃ¨res Ã  soupe d'huile d'olive")
          st.write("- Sel et poivre")
          # Charger l'image "bar_cuit"
          bar_cuit = cv2.imread("bar_cuit.jpg")
          # Afficher l'image "bar_cuit"
          st.image(bar_cuit[:, :, ::-1], caption='Bar cuit', use_column_width=True)
          st.write("Instructions :")
          st.write("1. PrÃ©chauffez le four Ã  200Â°C.")
          st.write("2. Lavez et sÃ©chez les citrons, puis coupez-les en rondelles.")
          st.write("3. Ãpluchez et hachez finement l'ail.")
          st.write("4. Lavez et sÃ©chez les branches de thym.")
          st.write("5. Dans un bol, mÃ©langez l'huile d'olive avec l'ail hachÃ© et les feuilles de thym. Salez et poivrez selon vos goÃ»ts.")
          st.write("6. DÃ©posez les rondelles de citron sur la plaque du four, puis posez le bar dessus.")
          st.write("7. Badigeonnez gÃ©nÃ©reusement le bar avec le mÃ©lange d'huile d'olive, d'ail et de thym.")
          st.write("8. Enfournez le bar pendant 25 Ã  30 minutes, jusqu'Ã  ce qu'il soit bien cuit et dorÃ©.")
          st.write("9. Servez le bar chaud, accompagnÃ© des rondelles de citron et Ã©ventuellement d'une salade verte.")

          st.write("Bon appÃ©tit !")
          
                    
        if label == 'Black Sea Sprat (Clupeonella)':  # VÃ©rifier si l'image est une classe de clupeonella
          
          st.markdown("<h3 style='text-align: center;'>Clupeonella:</h3>", unsafe_allow_html=True)
          
          clupe = cv2.imread("clupe.png")
          st.write("Clupeonella est un genre de petits poissons de la famille des Clupeidae, Ã©galement connus sous le nom de sprats ou d'Ã©perlan. Ils sont souvent utilisÃ©s pour prÃ©parer des plats de fruits de mer, notamment en Europe de l'Est et en Scandinavie, et sont souvent consommÃ©s fumÃ©s, salÃ©s, marinÃ©s ou en conserve.")
          st.write("# Fiche technique sur la Clupeonella")
          st.write("- Nom scientifique : Clupeonella")
          st.write("- Famille : Clupeidae")
          st.write("- Ordre : Clupeiformes")
          st.write("- Habitat : La Clupeonella est un poisson pÃ©lagique, ce qui signifie qu'elle vit dans les eaux libres de la mer, de l'ocÃ©an ou des lacs. Elle est principalement prÃ©sente dans les eaux douces et saumÃ¢tres de l'Europe et de l'Asie, bien qu'elle puisse Ã©galement Ãªtre trouvÃ©e en mer Noire et en MÃ©diterranÃ©e.")
          st.write("- Alimentation : Les Clupeonella se nourrissent principalement de zooplancton et de petits poissons.")
          st.write("- Taille : La taille de la Clupeonella varie en fonction de l'espÃ¨ce, mais en gÃ©nÃ©ral, elle mesure entre 5 et 15 centimÃ¨tres.")
          st.write("- Apparence : La Clupeonella a une apparence typique des poissons de la famille des Clupeidae, avec un corps fuselÃ©, argentÃ© et allongÃ©. Elle a une nageoire dorsale courte et une nageoire anale longue.")
          st.write("- Reproduction : La Clupeonella se reproduit en groupe. Les femelles pondent leurs Åufs dans les eaux libres, qui Ã©closent aprÃ¨s quelques jours. Les alevins sont pÃ©lagiques et se nourrissent de zooplancton jusqu'Ã  ce qu'ils atteignent leur taille adulte.")
          st.write("- Importance Ã©conomique : La Clupeonella est une espÃ¨ce de poisson importante pour la pÃªche commerciale et la consommation humaine. Elle est Ã©galement utilisÃ©e comme appÃ¢t pour la pÃªche sportive.")

          
          st.write("# Recette pour les sprats marinÃ©s")
          st.write("IngrÃ©dients :")
          st.write("- 500g de sprats")
          st.write("- 1/2 tasse de vinaigre blanc")
          st.write("- 1/2 tasse d'eau")
          st.write("- 1/4 tasse d'huile d'olive")
          st.write("- 1 oignon rouge Ã©mincÃ©")
          st.write("- 2 gousses d'ail Ã©mincÃ©es")
          st.write("- 1 cuillÃ¨re Ã  soupe de graines de coriandre")
          st.write("- 1 cuillÃ¨re Ã  soupe de gros sel de mer")
          st.write("- Poivre noir moulu")
          st.write("- Persil frais hachÃ© pour garnir")
          st.image(clupe[:, :, ::-1], caption='ClupÃ©onelle cuite', use_column_width=True)
          
          st.write("Instructions :")
          st.write("1. Lavez les sprats Ã  l'eau froide, puis sÃ©chez-les avec du papier absorbant.")
          st.write("2. Dans un grand bol, mÃ©langez le vinaigre blanc, l'eau, l'huile d'olive, l'oignon rouge, l'ail, les graines de coriandre, le sel et le poivre noir moulu.")
          st.write("3. Ajoutez les sprats dans le bol et assurez-vous qu'ils sont bien recouverts de la marinade. Couvrez le bol de film plastique et laissez mariner au rÃ©frigÃ©rateur pendant au moins 1 heure, voire toute la nuit.")
          st.write("4. PrÃ©chauffez le four Ã  200Â°C.")
          st.write("5. Ãgouttez les sprats de la marinade et dÃ©posez-les sur une plaque de cuisson recouverte de papier sulfurisÃ©.")
          st.write("6. Faites cuire les sprats au four pendant 10 Ã  12 minutes, jusqu'Ã  ce qu'ils soient dorÃ©s et croustillants.")
          st.write("7. Servez les sprats chauds, garnis de persil frais hachÃ©.")

          st.write("Bon appÃ©tit !")
        if label == 'Gilt-Head Bream (Dorade royale)':
            st.markdown("<h3 style='text-align: center;'>Dorade royale:</h3>", unsafe_allow_html=True)
            st.write("La dorade royale, Ã©galement connue sous le nom de dorade commune, est un poisson marin de la famille des SparidÃ©s, prÃ©sent dans les eaux chaudes de la MÃ©diterranÃ©e et de l'Atlantique. Ce poisson est apprÃ©ciÃ© pour sa chair blanche et ferme, ainsi que pour sa saveur dÃ©licate.")
            st.write("La dorade royale est un poisson trÃ¨s prisÃ© en cuisine, souvent prÃ©parÃ© entier et grillÃ© au four ou Ã  la plancha. Sa chair peut Ã©galement Ãªtre utilisÃ©e pour la prÃ©paration de soupes, de ragoÃ»ts ou de currys. Sa peau argentÃ©e et brillante est Ã©galement comestible et peut Ãªtre grillÃ©e pour devenir croustillante et savoureuse.")
            st.write("En plus d'Ãªtre savoureuse, la dorade royale est Ã©galement riche en nutriments bÃ©nÃ©fiques pour la santÃ©, tels que des acides gras omÃ©ga-3, des vitamines B12 et D, ainsi que des minÃ©raux comme le sÃ©lÃ©nium et le phosphore.")
            st.write("Cependant, il est important de choisir des dorades royales issues de la pÃªche durable, car cette espÃ¨ce est souvent surexploitÃ©e en raison de sa grande popularitÃ©. Il est donc recommandÃ© de choisir des poissons certifiÃ©s MSC (Marine Stewardship Council) ou d'autres certifications de durabilitÃ© pour s'assurer que l'on consomme de la dorade royale pÃªchÃ©e de maniÃ¨re responsable.")
          
            st.write("# Fiche technique sur la Dorade royale")
            st.write("- Nom scientifique : Sparus aurata")
            st.write("- Famille : Sparidae")
            st.write("- Habitat : La Dorade royale vit dans les eaux cÃ´tiÃ¨res peu profondes de la MÃ©diterranÃ©e et de l'Atlantique Est. Elle se trouve souvent prÃ¨s des rÃ©cifs, des Ã©paves de navires et des zones rocheuses.")
            st.write("- Alimentation : La Dorade royale se nourrit principalement de petits poissons, de crustacÃ©s et de mollusques.")
            st.write("- Taille : La taille moyenne de la Dorade royale est d'environ 30 cm, bien qu'elle puisse atteindre une longueur maximale de 70 cm.")
            st.write("- Apparence : La Dorade royale a un corps ovale et compressÃ© latÃ©ralement. Sa couleur varie du gris-argentÃ© au dorÃ© et elle a des taches dorÃ©es sur les joues et les opercules branchiaux. Elle a une nageoire dorsale unique et deux nageoires anales.")
            st.write("- Reproduction : La Dorade royale se reproduit pendant les mois d'Ã©tÃ©. Les femelles pondent des Åufs dans les eaux peu profondes, oÃ¹ ils Ã©closent aprÃ¨s quelques jours. Les alevins sont pÃ©lagiques et se nourrissent de zooplancton jusqu'Ã  ce qu'ils atteignent leur taille adulte.")
            st.write("- Importance Ã©conomique : La Dorade royale est une espÃ¨ce de poisson importante pour la pÃªche commerciale et la consommation humaine. Elle est Ã©galement une espÃ¨ce populaire pour la pÃªche sportive.")

            st.write("# Recette de la Dorade royale grillÃ©e")
            st.write("La Dorade royale grillÃ©e est une recette simple et dÃ©licieuse qui met en valeur les saveurs du poisson. Voici les Ã©tapes pour prÃ©parer cette recette :")
            
            # IngrÃ©dients
            st.write("## IngrÃ©dients :")
            st.write("- 1 Dorade royale entiÃ¨re, vidÃ©e et Ã©caillÃ©e")
            st.write("- 2 citrons")
            st.write("- 2 gousses d'ail, hachÃ©es finement")
            st.write("- 2 cuillÃ¨res Ã  soupe d'huile d'olive")
            st.write("- 1 cuillÃ¨re Ã  soupe de thym frais, hachÃ©")
            st.write("- Sel et poivre")
            
            clupe = cv2.imread("dorade_grille.jpg")
            st.image(clupe[:, :, ::-1], caption='Dorade cuite', use_column_width=True)
            
            # PrÃ©paration
            st.write("## PrÃ©paration :")
            st.write("1. PrÃ©chauffez le grill Ã  feu moyen-vif.")
            st.write("2. Coupez les citrons en rondelles et placez-les dans le ventre de la Dorade.")
            st.write("3. Dans un petit bol, mÃ©langez l'ail, l'huile d'olive, le thym, le sel et le poivre.")
            st.write("4. Badigeonnez le mÃ©lange sur la Dorade, en vous assurant de couvrir toute la surface.")
            st.write("5. Placez la Dorade sur la grille du grill et faites cuire pendant environ 6-8 minutes de chaque cÃ´tÃ©, en retournant une fois.")
            st.write("6. Servez immÃ©diatement avec les quartiers de citron et un peu de persil frais pour garnir.")
            
            st.write("Bon appÃ©tit !")
        if label == 'Hourse Mackerel (Chinchard)':
            st.markdown("<h3 style='text-align: center;'>Chinchard:</h3>", unsafe_allow_html=True)
            st.write("Le Chinchard, Ã©galement connu sous le nom de saurel, est un poisson de la famille des Carangidae, que l'on trouve dans les eaux cÃ´tiÃ¨res de l'Atlantique, du Pacifique et de l'ocÃ©an Indien. Ce poisson a un corps fusiforme et argentÃ©, avec des rayures verticales foncÃ©es sur les flancs.")
            st.write("Le Chinchard est un poisson trÃ¨s rapide et agile, capable de nager Ã  des vitesses Ã©levÃ©es grÃ¢ce Ã  sa nageoire caudale Ã©nergique. Il est souvent associÃ© Ã  des bancs de sardines et d'anchois, qu'il chasse en nageant rapidement Ã  travers les bancs.")
            st.write("Le Chinchard est Ã©galement un poisson savoureux et nutritif, riche en protÃ©ines et en acides gras omÃ©ga-3. Il peut Ãªtre prÃ©parÃ© de nombreuses faÃ§ons diffÃ©rentes, notamment grillÃ©, cuit au four ou en papillote.")
            st.write("En raison de sa popularitÃ© pour la pÃªche sportive et la consommation humaine, le Chinchard est une espÃ¨ce de poisson importante pour la pÃªche commerciale dans de nombreuses rÃ©gions du monde. Cependant, comme pour de nombreuses autres espÃ¨ces de poissons, il est important de gÃ©rer les stocks de Chinchard de maniÃ¨re durable pour assurer leur survie Ã  long terme.")
            
            st.write("## Fiche technique :")
            st.write("- Nom scientifique : Trachurus trachurus")
            st.write("- Famille : Carangidae")
            st.write("- Habitat : Le Chinchard se trouve dans les eaux cÃ´tiÃ¨res peu profondes de l'Atlantique, du Pacifique et de l'ocÃ©an Indien, ainsi que dans la MÃ©diterranÃ©e. Il est souvent associÃ© Ã  des bancs de sardines et d'anchois.")
            st.write("- Alimentation : Le Chinchard se nourrit principalement de petits poissons, de crustacÃ©s et de cÃ©phalopodes.")
            st.write("- Taille : La taille moyenne du Chinchard est d'environ 30 cm, bien qu'il puisse atteindre une longueur maximale de 70 cm.")
            st.write("- Reproduction : La pÃ©riode de reproduction du Chinchard varie selon la rÃ©gion et la tempÃ©rature de l'eau. Les femelles pondent des Åufs en eau libre, qui Ã©closent aprÃ¨s quelques jours. Les larves dÃ©rivantes se nourrissent de zooplancton jusqu'Ã  ce qu'elles atteignent leur taille adulte.")
            st.write("- Importance Ã©conomique : Le Chinchard est une espÃ¨ce de poisson importante pour la pÃªche commerciale et la consommation humaine. Il est Ã©galement une espÃ¨ce populaire pour la pÃªche sportive.")
            
            st.write("# Recette de Chinchard grillÃ© au citron et aux herbes")
            # IngrÃ©dients
            st.write("## IngrÃ©dients :")
            st.write("- 4 filets de Chinchard")
            st.write("- 1 citron")
            st.write("- 2 gousses d'ail")
            st.write("- 2 cuillÃ¨res Ã  soupe d'huile d'olive")
            st.write("- 1 cuillÃ¨re Ã  soupe de persil frais hachÃ©")
            st.write("- 1 cuillÃ¨re Ã  soupe de thym frais hachÃ©")
            st.write("- Sel et poivre")
            
            clupe = cv2.imread("chinchard_cuit.jpg")
            st.image(clupe[:, :, ::-1], caption='Chinchard cuit', use_column_width=True)
            
            # Instructions
            st.write("## Instructions :")
            st.write("1. PrÃ©chauffez le grill du four Ã  220Â°C.")
            st.write("2. Rincez les filets de Chinchard sous l'eau froide et sÃ©chez-les avec du papier absorbant.")
            st.write("3. Placez les filets de Chinchard sur une plaque de cuisson recouverte de papier d'aluminium.")
            st.write("4. Pressez le jus d'un citron sur les filets de Chinchard.")
            st.write("5. Ãmincez finement l'ail et saupoudrez-le sur les filets de Chinchard.")
            st.write("6. Ajoutez l'huile d'olive sur les filets de Chinchard.")
            st.write("7. Saupoudrez le persil et le thym frais sur les filets de Chinchard.")
            st.write("8. Salez et poivrez les filets de Chinchard selon votre goÃ»t.")
            st.write("9. Placez la plaque de cuisson sous le grill et faites cuire pendant environ 10 Ã  12 minutes, ou jusqu'Ã  ce que le poisson soit cuit et dorÃ©.")
            
            # Astuce
            st.write("## Astuce :")
            st.write("Servir le Chinchard grillÃ© avec une salade de roquette et de tomates fraÃ®ches pour un repas lÃ©ger et savoureux.")

            st.write("Bon appÃ©tit !")
        if label == 'Pas de poisson dÃ©tectÃ©':
            
            st.write("# DÃ©solÃ©, nous n'avons pas dÃ©tectÃ© de poisson dans cette image.")
            st.write("Nous sommes dÃ©solÃ©s, mais nous n'avons pas Ã©tÃ© en mesure de dÃ©tecter de poisson dans cette image. Cela peut Ãªtre dÃ» Ã  diffÃ©rents facteurs, tels que la qualitÃ© de l'image ou le fait qu'il n'y ait tout simplement pas de poisson prÃ©sent.")
            st.write("Nous vous recommandons de vÃ©rifier Ã  nouveau l'image et de vous assurer qu'elle est de bonne qualitÃ© et qu'elle contient bien un poisson. Si vous rencontrez toujours des difficultÃ©s, n'hÃ©sitez pas Ã  nous contacter pour obtenir de l'aide.")
            
        if label == 'Red Mullet (Rouget)':
            
            st.markdown("<h3 style='text-align: center;'>Rouget:</h3>", unsafe_allow_html=True)
            st.write("Le Rouget, Ã©galement connu sous le nom de Rouget-barbet, est un poisson dÃ©licieux et trÃ¨s apprÃ©ciÃ© dans la cuisine mÃ©diterranÃ©enne. Il est souvent servi grillÃ© ou cuit au four, avec une garniture de lÃ©gumes frais et d'herbes aromatiques.")
            st.write("Le Rouget est un poisson relativement petit, avec une chair ferme et savoureuse qui se marie bien avec une variÃ©tÃ© d'Ã©pices et de saveurs. Il est souvent pÃªchÃ© dans les eaux chaudes de la MÃ©diterranÃ©e, oÃ¹ il est une composante importante de la cuisine locale.")
            st.write("En plus d'Ãªtre savoureux, le Rouget est Ã©galement riche en nutriments importants tels que les protÃ©ines, les vitamines et les minÃ©raux. Il est Ã©galement faible en gras saturÃ©s, ce qui en fait un choix sain pour ceux qui cherchent Ã  maintenir une alimentation Ã©quilibrÃ©e.")
            st.write("Que vous soyez un fan de fruits de mer ou que vous cherchiez simplement Ã  dÃ©couvrir de nouveaux plats, le Rouget est un excellent choix pour ajouter une touche de saveur et de variÃ©tÃ© Ã  votre cuisine.")
            
            
            st.write("# Fiche technique : Le Rouget")
            
            st.write("## CaractÃ©ristiques")
            st.write("- Longueur moyenne : 15-30 cm")
            st.write("- Poids moyen : 200-300 g")
            st.write("- Couleur : Rouge vif sur le dos, avec un ventre argentÃ©")
            st.write("- Corps : Plat et ovale, avec une grande tÃªte et des yeux proÃ©minents")
            st.write("- RÃ©gime alimentaire : Carnivore, se nourrit principalement de crustacÃ©s et de petits poissons")
            
            st.write("## Utilisation culinaire")
            st.write("Le Rouget est un poisson trÃ¨s apprÃ©ciÃ© dans la cuisine mÃ©diterranÃ©enne. Il est souvent grillÃ© ou cuit au four, avec une garniture de lÃ©gumes frais et d'herbes aromatiques. Il peut Ã©galement Ãªtre utilisÃ© dans les soupes et les ragoÃ»ts de poisson.")
            
            st.write("## Valeur nutritive")
            st.write("- ProtÃ©ines : 19 g pour 100 g de poisson")
            st.write("- Lipides : 3 g pour 100 g de poisson")
            st.write("- Calories : 100 pour 100 g de poisson")
            st.write("- Nutriments : Riche en vitamines B6 et B12, en niacine et en sÃ©lÃ©nium")
            
            st.write("# Recette : Rouget grillÃ© avec lÃ©gumes provenÃ§aux")

            st.write("## IngrÃ©dients")
            st.write("- 4 Rougets")
            st.write("- 2 courgettes")
            st.write("- 2 tomates")
            st.write("- 1 oignon")
            st.write("- 2 gousses d'ail")
            st.write("- 1 citron")
            st.write("- Huile d'olive")
            st.write("- Sel")
            st.write("- Poivre")
            
            clupe = cv2.imread("rouget cuit.jpg")
            st.image(clupe[:, :, ::-1], caption='Rouget cuit', use_column_width=True)
            
            st.write("## PrÃ©paration")
            st.write("1. Lavez les lÃ©gumes et coupez-les en rondelles.")
            st.write("2. Ãpluchez l'oignon et coupez-le en fines lamelles.")
            st.write("3. Ãpluchez et hachez l'ail.")
            st.write("4. Faites chauffer un peu d'huile d'olive dans une poÃªle et faites revenir l'oignon et l'ail jusqu'Ã  ce qu'ils soient dorÃ©s.")
            st.write("5. Ajoutez les lÃ©gumes dans la poÃªle et faites-les cuire Ã  feu moyen pendant environ 10 minutes, en remuant rÃ©guliÃ¨rement.")
            st.write("6. Pendant ce temps, nettoyez les Rougets en enlevant les Ã©cailles et les tripes, et rincez-les sous l'eau froide.")
            st.write("7. Badigeonnez les Rougets avec de l'huile d'olive et du jus de citron.")
            st.write("8. Faites chauffer un grill ou une poÃªle antiadhÃ©sive et faites griller les Rougets pendant environ 5 minutes de chaque cÃ´tÃ©.")
            st.write("9. Assaisonnez les lÃ©gumes avec du sel et du poivre.")
            st.write("10. Servez les Rougets grillÃ©s avec les lÃ©gumes provenÃ§aux.")
            
            st.write("Bon appÃ©tit !")
            
        if label == 'Red Sea Bream (Dorade rose)':
            st.markdown("<h3 style='text-align: center;'>Dorade rose:</h3>", unsafe_allow_html=True)
            
            st.write("La Dorade rose, Ã©galement appelÃ©e Dorade royale rose ou Dorade rose de MÃ©diterranÃ©e, est un poisson de la famille des SparidÃ©s. Elle est prÃ©sente dans les eaux chaudes de la MÃ©diterranÃ©e, de l'Atlantique Est et du Sud-Ouest de l'Afrique. La Dorade rose peut atteindre une taille maximale d'environ 70 centimÃ¨tres de longueur et peut peser jusqu'Ã  6 kilogrammes.")
            st.write("La Dorade rose est un poisson apprÃ©ciÃ© pour sa chair fine, dÃ©licate et savoureuse. Elle se distingue des autres Dorades par sa couleur rose argentÃ©e, avec des reflets dorÃ©s sur les flancs. Ce poisson est Ã©galement riche en nutriments, notamment en protÃ©ines et en acides gras omÃ©ga-3.")
            st.write("La Dorade rose est pÃªchÃ©e dans les eaux de la MÃ©diterranÃ©e, principalement en France, en Espagne et en Italie. Elle est souvent consommÃ©e grillÃ©e ou cuite au four, accompagnÃ©e d'herbes aromatiques, de lÃ©gumes et d'huile d'olive. C'est un poisson de choix pour les amateurs de cuisine mÃ©diterranÃ©enne et les amateurs de poisson en gÃ©nÃ©ral.")
            
            st.write("# Fiche technique - Dorade rose")

            st.write("## CaractÃ©ristiques gÃ©nÃ©rales")
            st.write("- Nom scientifique : Sparus aurata")
            st.write("- Famille : SparidÃ©s")
            st.write("- Taille maximale : environ 70 centimÃ¨tres")
            st.write("- Poids maximal : jusqu'Ã  6 kilogrammes")
            st.write("- Habitat : eaux chaudes de la MÃ©diterranÃ©e, de l'Atlantique Est et du Sud-Ouest de l'Afrique")
            st.write("- Couleur : rose argentÃ©e avec des reflets dorÃ©s sur les cÃ´tÃ©s")
            
            st.write("## Nutrition")
            st.write("- ProtÃ©ines : environ 18 grammes pour 100 grammes de chair")
            st.write("- Lipides : environ 3 grammes pour 100 grammes de chair")
            st.write("- Acides gras omÃ©ga-3 : environ 500 milligrammes pour 100 grammes de chair")
            st.write("- Autres nutriments : vitamine B12, sÃ©lÃ©nium, iode, fer, magnÃ©sium")
            
            st.write("# Recette - Dorade rose grillÃ©e")

            st.write("## IngrÃ©dients")
            st.write("- 2 Dorades roses (environ 500 grammes chacune), vidÃ©es et nettoyÃ©es")
            st.write("- 2 citrons")
            st.write("- 4 gousses d'ail, hachÃ©es")
            st.write("- 4 cuillÃ¨res Ã  soupe d'huile d'olive")
            st.write("- Sel et poivre")
            st.write("- Herbes fraÃ®ches (au choix) : thym, romarin, persil")
            
            clupe = cv2.imread("dorade_cuit.jpg")
            st.image(clupe[:, :, ::-1], caption='Dorade cuite', use_column_width=True)
            
            st.write("## PrÃ©paration")
            st.write("1. PrÃ©chauffez le grill du four.")
            st.write("2. Rincez les Dorades roses Ã  l'eau froide et essuyez-les avec du papier absorbant.")
            st.write("3. Incisez chaque poisson sur le dos trois fois de chaque cÃ´tÃ©.")
            st.write("4. Salez et poivrez l'intÃ©rieur et l'extÃ©rieur des poissons.")
            st.write("5. Placez-les dans un plat allant au four.")
            st.write("6. Dans un petit bol, mÃ©langez l'huile d'olive, le jus de citron et l'ail hachÃ©.")
            st.write("7. Badigeonnez le mÃ©lange sur les poissons.")
            st.write("8. Ajoutez les herbes fraÃ®ches sur les poissons.")
            st.write("9. Enfournez et faites griller pendant environ 10 Ã  12 minutes, jusqu'Ã  ce que la peau soit croustillante et dorÃ©e.")
            st.write("10. Servez les Dorades roses grillÃ©es chaudes avec des quartiers de citron et des lÃ©gumes grillÃ©s.")

            st.write("Bon appÃ©tit !")
        
        if label == 'Shrimp (Crevette)':
            st.markdown("<h3 style='text-align: center;'>Crevette:</h3>", unsafe_allow_html=True)
            st.write("Les crevettes sont des crustacÃ©s marins appartenant Ã  la famille des PÃ©nÃ©idÃ©s. Elles sont caractÃ©risÃ©es par leur corps allongÃ© et leur carapace dure. Les crevettes sont l'un des fruits de mer les plus populaires dans le monde, en raison de leur goÃ»t dÃ©licat et de leur polyvalence culinaire.")
            st.write("Les crevettes sont riches en protÃ©ines, faibles en gras et contiennent des vitamines et des minÃ©raux tels que la vitamine D, le zinc et le sÃ©lÃ©nium. Elles sont Ã©galement une source importante d'acides gras omÃ©ga-3, qui sont bÃ©nÃ©fiques pour la santÃ© cardiaque et le fonctionnement du cerveau.")
            st.write("Il existe de nombreuses espÃ¨ces de crevettes, allant de la petite crevette rose commune aux crevettes gÃ©antes de la famille des PandalidÃ©s. Les crevettes sont utilisÃ©es dans de nombreuses cuisines Ã  travers le monde, telles que la cuisine asiatique, mÃ©diterranÃ©enne et crÃ©ole. Elles peuvent Ãªtre consommÃ©es crues, cuites, grillÃ©es, poÃªlÃ©es ou mÃªme frites.")
            
            st.write("## Fiche technique : la crevette")

            st.write("### CaractÃ©ristiques gÃ©nÃ©rales")
            st.write("- Nom scientifique : PÃ©nÃ©idÃ©s")
            st.write("- RÃ©gime alimentaire : omnivore")
            st.write("- Habitat : eau de mer")
            
            st.write("### Description physique")
            st.write("- Corps allongÃ©")
            st.write("- Carapace dure")
            st.write("- Antennes longues")
            st.write("- Pattes fines et grÃªles")
            
            st.write("### Informations nutritionnelles")
            st.write("- ProtÃ©ines : 18g pour 100g")
            st.write("- Glucides : 0g pour 100g")
            st.write("- Lipides : 0.8g pour 100g")
            st.write("- Vitamines : B12, D, E")
            st.write("- MinÃ©raux : sÃ©lÃ©nium, zinc, cuivre")
            
            st.write("### Utilisation en cuisine")
            st.write("- Consommation crue ou cuite")
            st.write("- Cuisson rapide (2 Ã  3 minutes)")
            st.write("- UtilisÃ©e dans de nombreuses cuisines (asiatique, mÃ©diterranÃ©enne, crÃ©ole, etc.)")
            st.write("- Peut Ãªtre prÃ©parÃ©e grillÃ©e, poÃªlÃ©e, frite, etc.")
            
            st.write("### Conseils d'achat")
            st.write("- FraÃ®cheur : chair ferme et non collante")
            st.write("- Taille : selon la recette, choisir une taille appropriÃ©e (petite, moyenne, grande)")
            st.write("- Provenance : privilÃ©gier les crevettes issues de l'aquaculture ou de la pÃªche durable")
            
            st.write("### Conservation")
            st.write("- Au rÃ©frigÃ©rateur : 1 Ã  2 jours maximum")
            st.write("- Au congÃ©lateur : jusqu'Ã  6 mois (dÃ©congeler lentement au rÃ©frigÃ©rateur)")
            
            st.write("## Recette : Crevettes sautÃ©es Ã  l'ail et au citron")

            st.write("### IngrÃ©dients")
            st.write("- 500g de crevettes dÃ©cortiquÃ©es et dÃ©veinÃ©es")
            st.write("- 4 gousses d'ail hachÃ©es")
            st.write("- Le jus d'un citron")
            st.write("- 2 cuillÃ¨res Ã  soupe d'huile d'olive")
            st.write("- Sel et poivre noir moulu")
            st.write("- Persil frais hachÃ© pour la garniture")
            
            clupe = cv2.imread("crevette_cuit.jpg")
            st.image(clupe[:, :, ::-1], caption='Crevette cuite', use_column_width=True)
            
            st.write("### PrÃ©paration")
            st.write("1. Dans une poÃªle Ã  feu moyen, chauffer l'huile d'olive.")
            st.write("2. Ajouter l'ail hachÃ© et cuire jusqu'Ã  ce qu'il soit dorÃ© et parfumÃ©.")
            st.write("3. Ajouter les crevettes dans la poÃªle et cuire jusqu'Ã  ce qu'elles deviennent roses et fermes, environ 3-4 minutes.")
            st.write("4. Ajouter le jus de citron, le sel et le poivre dans la poÃªle et bien mÃ©langer.")
            st.write("5. Garnir de persil frais hachÃ© et servir immÃ©diatement.")
            
            st.write("Bon appÃ©tit !")
        
        if label == 'Striped Red Mullet (Rouget-barbet de roche)':
            st.markdown("<h3 style='text-align: center;'>Rouget-barbet de roche:</h3>", unsafe_allow_html=True)
            st.write("Le Rouget-barbet de roche, Ã©galement appelÃ© Rouget-barbet mÃ©diterranÃ©en, est un poisson de mer appartenant Ã  la famille des Mullidae. Il est prÃ©sent dans les eaux cÃ´tiÃ¨res de la MÃ©diterranÃ©e et de l'Atlantique Est, et est particuliÃ¨rement apprÃ©ciÃ© pour sa chair fine et savoureuse.")
            st.write("Le Rouget-barbet de roche se caractÃ©rise par sa robe rose-orangÃ©e, avec des reflets argentÃ©s sur le ventre. Il possÃ¨de Ã©galement des nageoires dorsales et anales pointues, ainsi que des barbillons sous la mÃ¢choire qui lui ont valu son nom.")
            
            st.write("## Fiche technique : Rouget-barbet de roche")

            st.write("### Nom commun : Rouget-barbet de roche")
            st.write("### Nom scientifique : Mullus barbatus")
            st.write("### Famille : Mullidae")
            st.write("### Taille : jusqu'Ã  30 cm")
            st.write("### Poids : jusqu'Ã  1 kg")
            st.write("### Habitat : eaux cÃ´tiÃ¨res de la MÃ©diterranÃ©e et de l'Atlantique Est")
            st.write("### Alimentation : crustacÃ©s, mollusques, petits poissons")
            st.write("### Mode de pÃªche : ligne, filet")
            st.write("### SaisonnalitÃ© : mai Ã  dÃ©cembre")
            st.write("### Consommation : grillÃ©, poÃªlÃ©, en carpaccio")
            st.write("### Valeur nutritionnelle (pour 100 g) :")
            st.write("- Calories : 93 kcal")
            st.write("- ProtÃ©ines : 18,8 g")
            st.write("- Glucides : 0 g")
            st.write("- Lipides : 1,5 g")
            st.write("- Vitamines : B3, B12, D")
            st.write("- MinÃ©raux : phosphore, potassium, magnÃ©sium, fer")
            
            st.write("## Recette : Rouget-barbet de roche en papillote")

            st.write("### IngrÃ©dients (pour 2 personnes) :")
            st.write("- 2 Rougets-barbets de roche")
            st.write("- 2 tomates")
            st.write("- 1 oignon")
            st.write("- 1 citron")
            st.write("- 2 branches de thym")
            st.write("- 2 branches de romarin")
            st.write("- Huile d'olive")
            st.write("- Sel et poivre")
            
            clupe = cv2.imread("rouget_barbet_cuit.jpg")
            st.image(clupe[:, :, ::-1], caption='rouget-barbet de roche cuit', use_column_width=True)
            
            st.write("### PrÃ©paration :")
            st.write("1. PrÃ©chauffer le four Ã  180Â°C.")
            st.write("2. Laver les tomates et les couper en rondelles.")
            st.write("3. Ãplucher l'oignon et le couper en rondelles.")
            st.write("4. Laver le citron et le couper en rondelles.")
            st.write("5. Couper les branches de thym et de romarin en petits morceaux.")
            st.write("6. Huiler lÃ©gÃ¨rement deux grandes feuilles de papier sulfurisÃ©.")
            st.write("7. Poser les Rougets-barbets de roche sur les feuilles de papier sulfurisÃ©.")
            st.write("8. Disposer les rondelles de tomates, d'oignon et de citron autour des poissons.")
            st.write("9. Saupoudrer de thym et de romarin.")
            st.write("10. Saler et poivrer.")
            st.write("11. Verser un filet d'huile d'olive sur les poissons.")
            st.write("12. Refermer les papillotes en ramenant les bords des feuilles de papier sulfurisÃ© vers le centre et en les pliant plusieurs fois.")
            st.write("13. Enfourner pendant 20 minutes.")
            
            st.write("### Bon appÃ©tit !")
        
        if label == 'Trout (Truite)':
            st.markdown("<h3 style='text-align: center;'>Truite:</h3>", unsafe_allow_html=True)
            st.write("La truite est un poisson d'eau douce appartenant Ã  la famille des salmonidÃ©s. Elle est souvent pÃªchÃ©e pour sa chair dÃ©licate et sa saveur lÃ©gÃ¨rement sucrÃ©e. La truite est prÃ©sente dans les riviÃ¨res, les lacs et les Ã©tangs du monde entier et se distingue par sa robe tachetÃ©e de couleurs variÃ©es allant du brun au vert en passant par le rouge et le rose.")
            st.write("Si vous envisagez de pÃªcher la truite, il est important de connaÃ®tre les rÃ¨gles et les rÃ©glementations en vigueur dans votre rÃ©gion. La truite Ã©tant un poisson trÃ¨s populaire, certaines zones peuvent Ãªtre rÃ©glementÃ©es et nÃ©cessiter un permis de pÃªche. Par ailleurs, la taille minimale de la truite autorisÃ©e Ã  la pÃªche peut varier selon les endroits et les saisons.")
            
            st.write("# Fiche technique de la truite")

            st.write("## CaractÃ©ristiques")
            st.write("- Taille moyenne : entre 20 et 50 cm")
            st.write("- Poids moyen : entre 0,5 et 2 kg")
            st.write("- EspÃ©rance de vie : de 3 Ã  8 ans")
            st.write("- Habitat : eaux douces, riviÃ¨res, lacs, Ã©tangs")
            
            st.write("## Nutrition")
            st.write("- Riche en protÃ©ines")
            st.write("- Bonne source d'acides gras omÃ©ga-3")
            st.write("- Faible teneur en matiÃ¨res grasses")
            
            st.write("# Recette de la truite grillÃ©e")

            st.write("## IngrÃ©dients")
            st.write("- 2 truites vidÃ©es")
            st.write("- 2 gousses d'ail hachÃ©es")
            st.write("- 2 cuillÃ¨res Ã  soupe d'huile d'olive")
            st.write("- 1 citron")
            st.write("- Sel et poivre")
            
            clupe = cv2.imread("truite_cuit.jpg")
            st.image(clupe[:, :, ::-1], caption='Truite cuite', use_column_width=True)
            
            st.write("## PrÃ©paration")
            st.write("1. PrÃ©chauffez le grill du four.")
            st.write("2. Dans un petit bol, mÃ©langez l'huile d'olive, l'ail hachÃ©, le jus de citron, le sel et le poivre.")
            st.write("3. Badigeonnez les truites avec le mÃ©lange d'huile d'olive.")
            st.write("4. Placez les truites sur la grille du four et faites cuire pendant environ 8 Ã  10 minutes de chaque cÃ´tÃ©, jusqu'Ã  ce qu'elles soient dorÃ©es.")
            st.write("5. Servez chaud avec des quartiers de citron et des lÃ©gumes verts.")
            st.write("### Bon appétit !")
            
# Lancer l'application
if __name__ == '__main__':
    app()
