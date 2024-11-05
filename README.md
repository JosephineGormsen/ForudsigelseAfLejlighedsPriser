Dette projekt forudsiger lejlighedspriser baseret på faktorer som kvadratmeter, etage, altan, antal værelser, byggeår og renoveringsstatus. Projektet demonstrerer anvendelsen af to machine learning-modeller i Python: Lineær Regression og Random Forest Regression.

Projektindhold
Data Generation: Et syntetisk datasæt genereres med 500 eksempler af lejligheder med funktioner som størrelse, etage, altan osv.
Feature Engineering & Splitting: Datasættet opdeles i features og målvariabel (pris), og data splittes i træning og test.

Modeller:
Lineær Regression: En simpel lineær model for at vise en lineær sammenhæng.
Random Forest Regression: En model baseret på beslutningstræer, der typisk fanger komplekse mønstre bedre.
Model Evaluation: Evaluerer begge modeller ved hjælp af Mean Absolute Error, Mean Squared Error, Root Mean Squared Error og R²-score.

Visualiseringer:
Scatter plot af kvadratmeter vs. pris med Lineær Regression.
Feature Importance for Random Forest-modellen.
Sammenligning af faktiske og forudsigte priser for begge modeller.
