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
labels = ['Black Sea Sprat (Clupeonella)', 'Gilt-Head Bream (Dorade royale)', 'Hourse Mackerel (Chinchard)', "Pas de poisson détecté", 'Red Mullet (Rouget)', 'Red Sea Bream (Dorade rose)', 'Sea Bass (Bar)', 'Shrimp (Crevette)', 'Striped Red Mullet (Rouget-barbet de roche)', 'Trout (Truite)']

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
          if st.button("# Fiche technique sur le bar"):
              st.write("Nom : Bar (Dicentrarchus labrax)")
              st.write("Apparence : Poisson blanc avec une peau argentée et des écailles sombres. La taille varie de 20 à 100 cm.")
              st.write("Habitat : Le bar est présent dans les eaux côtières et les estuaires de l'Atlantique, de la Méditerranée et de la mer Noire.")
              st.write("Régime alimentaire : Le bar se nourrit de petits poissons, de crustacés et de céphalopodes.")
              st.write("Valeur nutritive : Le bar est riche en protéines de haute qualité, en acides gras oméga-3 et en vitamines B12 et D.")
              st.write("Préparation culinaire : Le bar est souvent grillé ou cuit au four, mais peut également être préparé en tartare ou en ceviche.")
              st.write("Saisonnalité : Le bar est généralement disponible toute l'année, mais sa saison de pêche optimale se situe entre le printemps et l'automne.")
                      
          if st.button("# Recette de bar")
              st.write("# Recette de bar")
              st.write("Ingrédients :")
              st.write("- 1 bar entier vidé et écaillé")
              st.write("- 2 citrons")
              st.write("- 3 gousses d'ail")
              st.write("- 4 branches de thym frais")
              st.write("- 4 cuillères à soupe d'huile d'olive")
              st.write("- Sel et poivre")
              # Charger l'image "bar_cuit"
              bar_cuit = cv2.imread("bar_cuit.jpg")
              # Afficher l'image "bar_cuit"
              st.image(bar_cuit[:, :, ::-1], caption='Bar cuit', use_column_width=True)
              st.write("Instructions :")
              st.write("1. Préchauffez le four à 200°C.")
              st.write("2. Lavez et séchez les citrons, puis coupez-les en rondelles.")
              st.write("3. Épluchez et hachez finement l'ail.")
              st.write("4. Lavez et séchez les branches de thym.")
              st.write("5. Dans un bol, mélangez l'huile d'olive avec l'ail haché et les feuilles de thym. Salez et poivrez selon vos goûts.")
              st.write("6. Déposez les rondelles de citron sur la plaque du four, puis posez le bar dessus.")
              st.write("7. Badigeonnez généreusement le bar avec le mélange d'huile d'olive, d'ail et de thym.")
              st.write("8. Enfournez le bar pendant 25 à 30 minutes, jusqu'à ce qu'il soit bien cuit et doré.")
              st.write("9. Servez le bar chaud, accompagné des rondelles de citron et éventuellement d'une salade verte.")

              st.write("Bon appétit !")
          
                    
        if label == 'Black Sea Sprat (Clupeonella)':  # Vérifier si l'image est une classe de clupeonella
          
          st.markdown("<h3 style='text-align: center;'>Clupeonella:</h3>", unsafe_allow_html=True)
          
          clupe = cv2.imread("clupe.png")
          st.write("Clupeonella est un genre de petits poissons de la famille des Clupeidae, également connus sous le nom de sprats ou d'éperlan. Ils sont souvent utilisés pour préparer des plats de fruits de mer, notamment en Europe de l'Est et en Scandinavie, et sont souvent consommés fumés, salés, marinés ou en conserve.")
          if st.button("# Fiche technique sur la Clupeonella"):
              st.write("- Nom scientifique : Clupeonella")
              st.write("- Famille : Clupeidae")
              st.write("- Ordre : Clupeiformes")
              st.write("- Habitat : La Clupeonella est un poisson pélagique, ce qui signifie qu'elle vit dans les eaux libres de la mer, de l'océan ou des lacs. Elle est principalement présente dans les eaux douces et saumâtres de l'Europe et de l'Asie, bien qu'elle puisse également être trouvée en mer Noire et en Méditerranée.")
              st.write("- Alimentation : Les Clupeonella se nourrissent principalement de zooplancton et de petits poissons.")
              st.write("- Taille : La taille de la Clupeonella varie en fonction de l'espèce, mais en général, elle mesure entre 5 et 15 centimètres.")
              st.write("- Apparence : La Clupeonella a une apparence typique des poissons de la famille des Clupeidae, avec un corps fuselé, argenté et allongé. Elle a une nageoire dorsale courte et une nageoire anale longue.")
              st.write("- Reproduction : La Clupeonella se reproduit en groupe. Les femelles pondent leurs œufs dans les eaux libres, qui éclosent après quelques jours. Les alevins sont pélagiques et se nourrissent de zooplancton jusqu'à ce qu'ils atteignent leur taille adulte.")
              st.write("- Importance économique : La Clupeonella est une espèce de poisson importante pour la pêche commerciale et la consommation humaine. Elle est également utilisée comme appât pour la pêche sportive.")

          
          if st.button("# Recette pour les sprats marinés")
              st.write("# Recette pour les sprats marinés")
              st.write("Ingrédients :")
              st.write("- 500g de sprats")
              st.write("- 1/2 tasse de vinaigre blanc")
              st.write("- 1/2 tasse d'eau")
              st.write("- 1/4 tasse d'huile d'olive")
              st.write("- 1 oignon rouge émincé")
              st.write("- 2 gousses d'ail émincées")
              st.write("- 1 cuillère à soupe de graines de coriandre")
              st.write("- 1 cuillère à soupe de gros sel de mer")
              st.write("- Poivre noir moulu")
              st.write("- Persil frais haché pour garnir")
              st.image(clupe[:, :, ::-1], caption='Clupéonelle cuite', use_column_width=True)

              st.write("Instructions :")
              st.write("1. Lavez les sprats à l'eau froide, puis séchez-les avec du papier absorbant.")
              st.write("2. Dans un grand bol, mélangez le vinaigre blanc, l'eau, l'huile d'olive, l'oignon rouge, l'ail, les graines de coriandre, le sel et le poivre noir moulu.")
              st.write("3. Ajoutez les sprats dans le bol et assurez-vous qu'ils sont bien recouverts de la marinade. Couvrez le bol de film plastique et laissez mariner au réfrigérateur pendant au moins 1 heure, voire toute la nuit.")
              st.write("4. Préchauffez le four à 200°C.")
              st.write("5. Égouttez les sprats de la marinade et déposez-les sur une plaque de cuisson recouverte de papier sulfurisé.")
              st.write("6. Faites cuire les sprats au four pendant 10 à 12 minutes, jusqu'à ce qu'ils soient dorés et croustillants.")
              st.write("7. Servez les sprats chauds, garnis de persil frais haché.")

              st.write("Bon appétit !")
        if label == 'Gilt-Head Bream (Dorade royale)':
            st.markdown("<h3 style='text-align: center;'>Dorade royale:</h3>", unsafe_allow_html=True)
            st.write("La dorade royale, également connue sous le nom de dorade commune, est un poisson marin de la famille des Sparidés, présent dans les eaux chaudes de la Méditerranée et de l'Atlantique. Ce poisson est apprécié pour sa chair blanche et ferme, ainsi que pour sa saveur délicate.")
            st.write("La dorade royale est un poisson très prisé en cuisine, souvent préparé entier et grillé au four ou à la plancha. Sa chair peut également être utilisée pour la préparation de soupes, de ragoûts ou de currys. Sa peau argentée et brillante est également comestible et peut être grillée pour devenir croustillante et savoureuse.")
            st.write("En plus d'être savoureuse, la dorade royale est également riche en nutriments bénéfiques pour la santé, tels que des acides gras oméga-3, des vitamines B12 et D, ainsi que des minéraux comme le sélénium et le phosphore.")
            st.write("Cependant, il est important de choisir des dorades royales issues de la pêche durable, car cette espèce est souvent surexploitée en raison de sa grande popularité. Il est donc recommandé de choisir des poissons certifiés MSC (Marine Stewardship Council) ou d'autres certifications de durabilité pour s'assurer que l'on consomme de la dorade royale pêchée de manière responsable.")
          
            if st.button("# Fiche technique sur la Dorade royale"):
                st.write("- Nom scientifique : Sparus aurata")
                st.write("- Famille : Sparidae")
                st.write("- Habitat : La Dorade royale vit dans les eaux côtières peu profondes de la Méditerranée et de l'Atlantique Est. Elle se trouve souvent près des récifs, des épaves de navires et des zones rocheuses.")
                st.write("- Alimentation : La Dorade royale se nourrit principalement de petits poissons, de crustacés et de mollusques.")
                st.write("- Taille : La taille moyenne de la Dorade royale est d'environ 30 cm, bien qu'elle puisse atteindre une longueur maximale de 70 cm.")
                st.write("- Apparence : La Dorade royale a un corps ovale et compressé latéralement. Sa couleur varie du gris-argenté au doré et elle a des taches dorées sur les joues et les opercules branchiaux. Elle a une nageoire dorsale unique et deux nageoires anales.")
                st.write("- Reproduction : La Dorade royale se reproduit pendant les mois d'été. Les femelles pondent des œufs dans les eaux peu profondes, où ils éclosent après quelques jours. Les alevins sont pélagiques et se nourrissent de zooplancton jusqu'à ce qu'ils atteignent leur taille adulte.")
                st.write("- Importance économique : La Dorade royale est une espèce de poisson importante pour la pêche commerciale et la consommation humaine. Elle est également une espèce populaire pour la pêche sportive.")

            if st.button("# Recette de la Dorade royale grillée")
                st.write("# Recette de la Dorade royale grillée")
                st.write("La Dorade royale grillée est une recette simple et délicieuse qui met en valeur les saveurs du poisson. Voici les étapes pour préparer cette recette :")

                # Ingrédients
                st.write("## Ingrédients :")
                st.write("- 1 Dorade royale entière, vidée et écaillée")
                st.write("- 2 citrons")
                st.write("- 2 gousses d'ail, hachées finement")
                st.write("- 2 cuillères à soupe d'huile d'olive")
                st.write("- 1 cuillère à soupe de thym frais, haché")
                st.write("- Sel et poivre")

                clupe = cv2.imread("dorade_grille.jpg")
                st.image(clupe[:, :, ::-1], caption='Dorade cuite', use_column_width=True)

                # Préparation
                st.write("## Préparation :")
                st.write("1. Préchauffez le grill à feu moyen-vif.")
                st.write("2. Coupez les citrons en rondelles et placez-les dans le ventre de la Dorade.")
                st.write("3. Dans un petit bol, mélangez l'ail, l'huile d'olive, le thym, le sel et le poivre.")
                st.write("4. Badigeonnez le mélange sur la Dorade, en vous assurant de couvrir toute la surface.")
                st.write("5. Placez la Dorade sur la grille du grill et faites cuire pendant environ 6-8 minutes de chaque côté, en retournant une fois.")
                st.write("6. Servez immédiatement avec les quartiers de citron et un peu de persil frais pour garnir.")

                st.write("Bon appétit !")
        if label == 'Hourse Mackerel (Chinchard)':
            st.markdown("<h3 style='text-align: center;'>Chinchard:</h3>", unsafe_allow_html=True)
            st.write("Le Chinchard, également connu sous le nom de saurel, est un poisson de la famille des Carangidae, que l'on trouve dans les eaux côtières de l'Atlantique, du Pacifique et de l'océan Indien. Ce poisson a un corps fusiforme et argenté, avec des rayures verticales foncées sur les flancs.")
            st.write("Le Chinchard est un poisson très rapide et agile, capable de nager à des vitesses élevées grâce à sa nageoire caudale énergique. Il est souvent associé à des bancs de sardines et d'anchois, qu'il chasse en nageant rapidement à travers les bancs.")
            st.write("Le Chinchard est également un poisson savoureux et nutritif, riche en protéines et en acides gras oméga-3. Il peut être préparé de nombreuses façons différentes, notamment grillé, cuit au four ou en papillote.")
            st.write("En raison de sa popularité pour la pêche sportive et la consommation humaine, le Chinchard est une espèce de poisson importante pour la pêche commerciale dans de nombreuses régions du monde. Cependant, comme pour de nombreuses autres espèces de poissons, il est important de gérer les stocks de Chinchard de manière durable pour assurer leur survie à long terme.")
            
            if st.button("## Fiche technique :"):
                st.write("- Nom scientifique : Trachurus trachurus")
                st.write("- Famille : Carangidae")
                st.write("- Habitat : Le Chinchard se trouve dans les eaux côtières peu profondes de l'Atlantique, du Pacifique et de l'océan Indien, ainsi que dans la Méditerranée. Il est souvent associé à des bancs de sardines et d'anchois.")
                st.write("- Alimentation : Le Chinchard se nourrit principalement de petits poissons, de crustacés et de céphalopodes.")
                st.write("- Taille : La taille moyenne du Chinchard est d'environ 30 cm, bien qu'il puisse atteindre une longueur maximale de 70 cm.")
                st.write("- Reproduction : La période de reproduction du Chinchard varie selon la région et la température de l'eau. Les femelles pondent des œufs en eau libre, qui éclosent après quelques jours. Les larves dérivantes se nourrissent de zooplancton jusqu'à ce qu'elles atteignent leur taille adulte.")
                st.write("- Importance économique : Le Chinchard est une espèce de poisson importante pour la pêche commerciale et la consommation humaine. Il est également une espèce populaire pour la pêche sportive.")
            
            if st.button("# Recette de Chinchard grillé au citron et aux herbes")
                st.write("# Recette de Chinchard grillé au citron et aux herbes")
                # Ingrédients
                st.write("## Ingrédients :")
                st.write("- 4 filets de Chinchard")
                st.write("- 1 citron")
                st.write("- 2 gousses d'ail")
                st.write("- 2 cuillères à soupe d'huile d'olive")
                st.write("- 1 cuillère à soupe de persil frais haché")
                st.write("- 1 cuillère à soupe de thym frais haché")
                st.write("- Sel et poivre")

                clupe = cv2.imread("chinchard_cuit.jpg")
                st.image(clupe[:, :, ::-1], caption='Chinchard cuit', use_column_width=True)

                # Instructions
                st.write("## Instructions :")
                st.write("1. Préchauffez le grill du four à 220°C.")
                st.write("2. Rincez les filets de Chinchard sous l'eau froide et séchez-les avec du papier absorbant.")
                st.write("3. Placez les filets de Chinchard sur une plaque de cuisson recouverte de papier d'aluminium.")
                st.write("4. Pressez le jus d'un citron sur les filets de Chinchard.")
                st.write("5. Émincez finement l'ail et saupoudrez-le sur les filets de Chinchard.")
                st.write("6. Ajoutez l'huile d'olive sur les filets de Chinchard.")
                st.write("7. Saupoudrez le persil et le thym frais sur les filets de Chinchard.")
                st.write("8. Salez et poivrez les filets de Chinchard selon votre goût.")
                st.write("9. Placez la plaque de cuisson sous le grill et faites cuire pendant environ 10 à 12 minutes, ou jusqu'à ce que le poisson soit cuit et doré.")

                # Astuce
                st.write("## Astuce :")
                st.write("Servir le Chinchard grillé avec une salade de roquette et de tomates fraîches pour un repas léger et savoureux.")

                st.write("Bon appétit !")
        if label == 'Pas de poisson détecté':
            
            st.write("# Désolé, nous n'avons pas détecté de poisson dans cette image.")
            st.write("Nous sommes désolés, mais nous n'avons pas été en mesure de détecter de poisson dans cette image. Cela peut être dû à différents facteurs, tels que la qualité de l'image ou le fait qu'il n'y ait tout simplement pas de poisson présent.")
            st.write("Nous vous recommandons de vérifier à nouveau l'image et de vous assurer qu'elle est de bonne qualité et qu'elle contient bien un poisson. Si vous rencontrez toujours des difficultés, n'hésitez pas à nous contacter pour obtenir de l'aide.")
            
        if label == 'Red Mullet (Rouget)':
            
            st.markdown("<h3 style='text-align: center;'>Rouget:</h3>", unsafe_allow_html=True)
            st.write("Le Rouget, également connu sous le nom de Rouget-barbet, est un poisson délicieux et très apprécié dans la cuisine méditerranéenne. Il est souvent servi grillé ou cuit au four, avec une garniture de légumes frais et d'herbes aromatiques.")
            st.write("Le Rouget est un poisson relativement petit, avec une chair ferme et savoureuse qui se marie bien avec une variété d'épices et de saveurs. Il est souvent pêché dans les eaux chaudes de la Méditerranée, où il est une composante importante de la cuisine locale.")
            st.write("En plus d'être savoureux, le Rouget est également riche en nutriments importants tels que les protéines, les vitamines et les minéraux. Il est également faible en gras saturés, ce qui en fait un choix sain pour ceux qui cherchent à maintenir une alimentation équilibrée.")
            st.write("Que vous soyez un fan de fruits de mer ou que vous cherchiez simplement à découvrir de nouveaux plats, le Rouget est un excellent choix pour ajouter une touche de saveur et de variété à votre cuisine.")
            
            
            if st.button("# Fiche technique : Le Rouget"):
            
                st.write("## Caractéristiques")
                st.write("- Longueur moyenne : 15-30 cm")
                st.write("- Poids moyen : 200-300 g")
                st.write("- Couleur : Rouge vif sur le dos, avec un ventre argenté")
                st.write("- Corps : Plat et ovale, avec une grande tête et des yeux proéminents")
                st.write("- Régime alimentaire : Carnivore, se nourrit principalement de crustacés et de petits poissons")

                st.write("## Utilisation culinaire")
                st.write("Le Rouget est un poisson très apprécié dans la cuisine méditerranéenne. Il est souvent grillé ou cuit au four, avec une garniture de légumes frais et d'herbes aromatiques. Il peut également être utilisé dans les soupes et les ragoûts de poisson.")

                st.write("## Valeur nutritive")
                st.write("- Protéines : 19 g pour 100 g de poisson")
                st.write("- Lipides : 3 g pour 100 g de poisson")
                st.write("- Calories : 100 pour 100 g de poisson")
                st.write("- Nutriments : Riche en vitamines B6 et B12, en niacine et en sélénium")
            
            if st.button("# Recette : Rouget grillé avec légumes provençaux")
                st.write("# Recette : Rouget grillé avec légumes provençaux")

                st.write("## Ingrédients")
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

                st.write("## Préparation")
                st.write("1. Lavez les légumes et coupez-les en rondelles.")
                st.write("2. Épluchez l'oignon et coupez-le en fines lamelles.")
                st.write("3. Épluchez et hachez l'ail.")
                st.write("4. Faites chauffer un peu d'huile d'olive dans une poêle et faites revenir l'oignon et l'ail jusqu'à ce qu'ils soient dorés.")
                st.write("5. Ajoutez les légumes dans la poêle et faites-les cuire à feu moyen pendant environ 10 minutes, en remuant régulièrement.")
                st.write("6. Pendant ce temps, nettoyez les Rougets en enlevant les écailles et les tripes, et rincez-les sous l'eau froide.")
                st.write("7. Badigeonnez les Rougets avec de l'huile d'olive et du jus de citron.")
                st.write("8. Faites chauffer un grill ou une poêle antiadhésive et faites griller les Rougets pendant environ 5 minutes de chaque côté.")
                st.write("9. Assaisonnez les légumes avec du sel et du poivre.")
                st.write("10. Servez les Rougets grillés avec les légumes provençaux.")

                st.write("Bon appétit !")
            
        if label == 'Red Sea Bream (Dorade rose)':
            st.markdown("<h3 style='text-align: center;'>Dorade rose:</h3>", unsafe_allow_html=True)
            
            st.write("La Dorade rose, également appelée Dorade royale rose ou Dorade rose de Méditerranée, est un poisson de la famille des Sparidés. Elle est présente dans les eaux chaudes de la Méditerranée, de l'Atlantique Est et du Sud-Ouest de l'Afrique. La Dorade rose peut atteindre une taille maximale d'environ 70 centimètres de longueur et peut peser jusqu'à 6 kilogrammes.")
            st.write("La Dorade rose est un poisson apprécié pour sa chair fine, délicate et savoureuse. Elle se distingue des autres Dorades par sa couleur rose argentée, avec des reflets dorés sur les flancs. Ce poisson est également riche en nutriments, notamment en protéines et en acides gras oméga-3.")
            st.write("La Dorade rose est pêchée dans les eaux de la Méditerranée, principalement en France, en Espagne et en Italie. Elle est souvent consommée grillée ou cuite au four, accompagnée d'herbes aromatiques, de légumes et d'huile d'olive. C'est un poisson de choix pour les amateurs de cuisine méditerranéenne et les amateurs de poisson en général.")
            
            if st.button("# Fiche technique - Dorade rose"):

                st.write("## Caractéristiques générales")
                st.write("- Nom scientifique : Sparus aurata")
                st.write("- Famille : Sparidés")
                st.write("- Taille maximale : environ 70 centimètres")
                st.write("- Poids maximal : jusqu'à 6 kilogrammes")
                st.write("- Habitat : eaux chaudes de la Méditerranée, de l'Atlantique Est et du Sud-Ouest de l'Afrique")
                st.write("- Couleur : rose argentée avec des reflets dorés sur les côtés")

                st.write("## Nutrition")
                st.write("- Protéines : environ 18 grammes pour 100 grammes de chair")
                st.write("- Lipides : environ 3 grammes pour 100 grammes de chair")
                st.write("- Acides gras oméga-3 : environ 500 milligrammes pour 100 grammes de chair")
                st.write("- Autres nutriments : vitamine B12, sélénium, iode, fer, magnésium")
            
            if st.button("# Recette - Dorade rose grillée")
                st.write("# Recette - Dorade rose grillée")

                st.write("## Ingrédients")
                st.write("- 2 Dorades roses (environ 500 grammes chacune), vidées et nettoyées")
                st.write("- 2 citrons")
                st.write("- 4 gousses d'ail, hachées")
                st.write("- 4 cuillères à soupe d'huile d'olive")
                st.write("- Sel et poivre")
                st.write("- Herbes fraîches (au choix) : thym, romarin, persil")

                clupe = cv2.imread("dorade_cuit.jpg")
                st.image(clupe[:, :, ::-1], caption='Dorade cuite', use_column_width=True)

                st.write("## Préparation")
                st.write("1. Préchauffez le grill du four.")
                st.write("2. Rincez les Dorades roses à l'eau froide et essuyez-les avec du papier absorbant.")
                st.write("3. Incisez chaque poisson sur le dos trois fois de chaque côté.")
                st.write("4. Salez et poivrez l'intérieur et l'extérieur des poissons.")
                st.write("5. Placez-les dans un plat allant au four.")
                st.write("6. Dans un petit bol, mélangez l'huile d'olive, le jus de citron et l'ail haché.")
                st.write("7. Badigeonnez le mélange sur les poissons.")
                st.write("8. Ajoutez les herbes fraîches sur les poissons.")
                st.write("9. Enfournez et faites griller pendant environ 10 à 12 minutes, jusqu'à ce que la peau soit croustillante et dorée.")
                st.write("10. Servez les Dorades roses grillées chaudes avec des quartiers de citron et des légumes grillés.")

                st.write("Bon appétit !")
        
        if label == 'Shrimp (Crevette)':
            st.markdown("<h3 style='text-align: center;'>Crevette:</h3>", unsafe_allow_html=True)
            st.write("Les crevettes sont des crustacés marins appartenant à la famille des Pénéidés. Elles sont caractérisées par leur corps allongé et leur carapace dure. Les crevettes sont l'un des fruits de mer les plus populaires dans le monde, en raison de leur goût délicat et de leur polyvalence culinaire.")
            st.write("Les crevettes sont riches en protéines, faibles en gras et contiennent des vitamines et des minéraux tels que la vitamine D, le zinc et le sélénium. Elles sont également une source importante d'acides gras oméga-3, qui sont bénéfiques pour la santé cardiaque et le fonctionnement du cerveau.")
            st.write("Il existe de nombreuses espèces de crevettes, allant de la petite crevette rose commune aux crevettes géantes de la famille des Pandalidés. Les crevettes sont utilisées dans de nombreuses cuisines à travers le monde, telles que la cuisine asiatique, méditerranéenne et créole. Elles peuvent être consommées crues, cuites, grillées, poêlées ou même frites.")
            
            if st.button("## Fiche technique : la crevette"):

                st.write("### Caractéristiques générales")
                st.write("- Nom scientifique : Pénéidés")
                st.write("- Régime alimentaire : omnivore")
                st.write("- Habitat : eau de mer")

                st.write("### Description physique")
                st.write("- Corps allongé")
                st.write("- Carapace dure")
                st.write("- Antennes longues")
                st.write("- Pattes fines et grêles")

                st.write("### Informations nutritionnelles")
                st.write("- Protéines : 18g pour 100g")
                st.write("- Glucides : 0g pour 100g")
                st.write("- Lipides : 0.8g pour 100g")
                st.write("- Vitamines : B12, D, E")
                st.write("- Minéraux : sélénium, zinc, cuivre")

                st.write("### Utilisation en cuisine")
                st.write("- Consommation crue ou cuite")
                st.write("- Cuisson rapide (2 à 3 minutes)")
                st.write("- Utilisée dans de nombreuses cuisines (asiatique, méditerranéenne, créole, etc.)")
                st.write("- Peut être préparée grillée, poêlée, frite, etc.")

                st.write("### Conseils d'achat")
                st.write("- Fraîcheur : chair ferme et non collante")
                st.write("- Taille : selon la recette, choisir une taille appropriée (petite, moyenne, grande)")
                st.write("- Provenance : privilégier les crevettes issues de l'aquaculture ou de la pêche durable")

                st.write("### Conservation")
                st.write("- Au réfrigérateur : 1 à 2 jours maximum")
                st.write("- Au congélateur : jusqu'à 6 mois (décongeler lentement au réfrigérateur)")
            
            if st.button("## Recette : Crevettes sautées à l'ail et au citron")
                st.write("## Recette : Crevettes sautées à l'ail et au citron")

                st.write("### Ingrédients")
                st.write("- 500g de crevettes décortiquées et déveinées")
                st.write("- 4 gousses d'ail hachées")
                st.write("- Le jus d'un citron")
                st.write("- 2 cuillères à soupe d'huile d'olive")
                st.write("- Sel et poivre noir moulu")
                st.write("- Persil frais haché pour la garniture")

                clupe = cv2.imread("crevette_cuit.jpg")
                st.image(clupe[:, :, ::-1], caption='Crevette cuite', use_column_width=True)

                st.write("### Préparation")
                st.write("1. Dans une poêle à feu moyen, chauffer l'huile d'olive.")
                st.write("2. Ajouter l'ail haché et cuire jusqu'à ce qu'il soit doré et parfumé.")
                st.write("3. Ajouter les crevettes dans la poêle et cuire jusqu'à ce qu'elles deviennent roses et fermes, environ 3-4 minutes.")
                st.write("4. Ajouter le jus de citron, le sel et le poivre dans la poêle et bien mélanger.")
                st.write("5. Garnir de persil frais haché et servir immédiatement.")

                st.write("Bon appétit !")

        if label == 'Striped Red Mullet (Rouget-barbet de roche)':
            st.markdown("<h3 style='text-align: center;'>Rouget-barbet de roche:</h3>", unsafe_allow_html=True)
            st.write("Le Rouget-barbet de roche, également appelé Rouget-barbet méditerranéen, est un poisson de mer appartenant à la famille des Mullidae. Il est présent dans les eaux côtières de la Méditerranée et de l'Atlantique Est, et est particulièrement apprécié pour sa chair fine et savoureuse.")
            st.write("Le Rouget-barbet de roche se caractérise par sa robe rose-orangée, avec des reflets argentés sur le ventre. Il possède également des nageoires dorsales et anales pointues, ainsi que des barbillons sous la mâchoire qui lui ont valu son nom.")
            
            if st.button("## Fiche technique : Rouget-barbet de roche"):

                st.write("### Nom commun : Rouget-barbet de roche")
                st.write("### Nom scientifique : Mullus barbatus")
                st.write("### Famille : Mullidae")
                st.write("### Taille : jusqu'à 30 cm")
                st.write("### Poids : jusqu'à 1 kg")
                st.write("### Habitat : eaux côtières de la Méditerranée et de l'Atlantique Est")
                st.write("### Alimentation : crustacés, mollusques, petits poissons")
                st.write("### Mode de pêche : ligne, filet")
                st.write("### Saisonnalité : mai à décembre")
                st.write("### Consommation : grillé, poêlé, en carpaccio")
                st.write("### Valeur nutritionnelle (pour 100 g) :")
                st.write("- Calories : 93 kcal")
                st.write("- Protéines : 18,8 g")
                st.write("- Glucides : 0 g")
                st.write("- Lipides : 1,5 g")
                st.write("- Vitamines : B3, B12, D")
                st.write("- Minéraux : phosphore, potassium, magnésium, fer")
            
            if st.button("## Recette : Rouget-barbet de roche en papillote")
                st.write("## Recette : Rouget-barbet de roche en papillote")

                st.write("### Ingrédients (pour 2 personnes) :")
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

                st.write("### Préparation :")
                st.write("1. Préchauffer le four à 180°C.")
                st.write("2. Laver les tomates et les couper en rondelles.")
                st.write("3. Éplucher l'oignon et le couper en rondelles.")
                st.write("4. Laver le citron et le couper en rondelles.")
                st.write("5. Couper les branches de thym et de romarin en petits morceaux.")
                st.write("6. Huiler légèrement deux grandes feuilles de papier sulfurisé.")
                st.write("7. Poser les Rougets-barbets de roche sur les feuilles de papier sulfurisé.")
                st.write("8. Disposer les rondelles de tomates, d'oignon et de citron autour des poissons.")
                st.write("9. Saupoudrer de thym et de romarin.")
                st.write("10. Saler et poivrer.")
                st.write("11. Verser un filet d'huile d'olive sur les poissons.")
                st.write("12. Refermer les papillotes en ramenant les bords des feuilles de papier sulfurisé vers le centre et en les pliant plusieurs fois.")
                st.write("13. Enfourner pendant 20 minutes.")

                st.write("### Bon appétit !")
        
        if label == 'Trout (Truite)':
            st.markdown("<h3 style='text-align: center;'>Truite:</h3>", unsafe_allow_html=True)
            st.write("La truite est un poisson d'eau douce appartenant à la famille des salmonidés. Elle est souvent pêchée pour sa chair délicate et sa saveur légèrement sucrée. La truite est présente dans les rivières, les lacs et les étangs du monde entier et se distingue par sa robe tachetée de couleurs variées allant du brun au vert en passant par le rouge et le rose.")
            st.write("Si vous envisagez de pêcher la truite, il est important de connaître les règles et les réglementations en vigueur dans votre région. La truite étant un poisson très populaire, certaines zones peuvent être réglementées et nécessiter un permis de pêche. Par ailleurs, la taille minimale de la truite autorisée à la pêche peut varier selon les endroits et les saisons.")
            
            if st.button("# Fiche technique de la truite"):

                st.write("## Caractéristiques")
                st.write("- Taille moyenne : entre 20 et 50 cm")
                st.write("- Poids moyen : entre 0,5 et 2 kg")
                st.write("- Espérance de vie : de 3 à 8 ans")
                st.write("- Habitat : eaux douces, rivières, lacs, étangs")

                st.write("## Nutrition")
                st.write("- Riche en protéines")
                st.write("- Bonne source d'acides gras oméga-3")
                st.write("- Faible teneur en matières grasses")
            
            if st.button("# Recette de la truite grillée")
                st.write("# Recette de la truite grillée")

                st.write("## Ingrédients")
                st.write("- 2 truites vidées")
                st.write("- 2 gousses d'ail hachées")
                st.write("- 2 cuillères à soupe d'huile d'olive")
                st.write("- 1 citron")
                st.write("- Sel et poivre")

                clupe = cv2.imread("truite_cuit.jpg")
                st.image(clupe[:, :, ::-1], caption='Truite cuite', use_column_width=True)

                st.write("## Préparation")
                st.write("1. Préchauffez le grill du four.")
                st.write("2. Dans un petit bol, mélangez l'huile d'olive, l'ail haché, le jus de citron, le sel et le poivre.")
                st.write("3. Badigeonnez les truites avec le mélange d'huile d'olive.")
                st.write("4. Placez les truites sur la grille du four et faites cuire pendant environ 8 à 10 minutes de chaque côté, jusqu'à ce qu'elles soient dorées.")
                st.write("5. Servez chaud avec des quartiers de citron et des légumes verts.")
                st.write("### Bon appétit !")
            
# Lancer l'application
if __name__ == '__main__':
    app()
