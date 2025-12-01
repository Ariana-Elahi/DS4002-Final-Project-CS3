# DS4002 Final Project CS3 Read Me

---

## Software and Platform
The software used in this project was **R** and **Python**.  

**Necessary libraries for R include:**
- ggplot2  
- dplyr  

**Necessary packages for Python include:**
- pandas  
- string  
- scikit-learn  
- SciPy  
- matplotlib  
- seaborn  
- NumPy  
- collections  

  
The platforms used for this project were **Windows** and **Mac**.

---

## Documentation Map
- **Data**
  - `All-seasons.csv` — original data file  
  - `Metadata.md` — description of data  
  - `cleaned_dataset.csv` — cleaned data file ready to be used in modeling

- **Scripts Folder**
  - `character_classification.py` — model code
  - `Exploratory Analysis South Park.R` - initial exploratory code, generates top ten characters by number of lines and total number of lines per season 

- **Other Files/Folders**
  - Supplemental Materials - Contains 2 reading materials to help with conceptual understanding of the topic and technical understanding of the topic.  
  - Hook - Explains the role of the recreater in the project.
  - README - Read me for the project
  - Rubric  - Guides the recreater through understanding and creating the project. 

---

## Instructions for Reproduction
### To run the exploratory analysis script:
1. Download the `All-seasons.csv` file to a known location on your computer. Download the `Exploratory Analysis South Park.R` file to the same location.
2. Run the `Exploratory Analysis South Park.R` file. Ensure that the file directory in your IDE is set to the location of `All-seasons.csv`.
3. Plots for the top ten characters by number of lines and total number of lines per season will be generated.
### To run the character classification script:
1. Download the `All-seasons.csv` file to a known location on your computer. Download the `character_classification.py` file to the same location.  
2. Run the `character_classification.py` file section by section. Ensure that the file directory in your IDE is set to the location of the `All-seasons.csv` and `character_classification.py`.  
3. Plots will be generated in the IDE. A `cleaned_dataset.csv` file will be generated containing the data ready for use in the model. The precision, recall, and F1 score table will be printed in the IDE console, as well as the top words and phrases for each of the top ten characters.
### Once the project scipts have been recreated these are the outputs you should be able to obtain:
  - `Confusion_Matrix.png` — confusion matrix from model  
  - `Most_Common_Misclassifications.png` — most common mistakes between the top 10 characters  
  - `Per_Class_Accuracy.png` — how accurate the model is for each of the top 10 characters  
  - `Precision_Recall_and_F1_by_Character.png` — precision, recall, and F1 score for each of the top 10 characters  
  - `Top_Words_Phrase_for_Each_Character.pdf` — most common words and phrases spoken by each of the top 10 characters  
  - `Top_10_Characters_by_Number_of_Lines.png` — top 10 characters ranked by number of lines  
  - `Total_Number_of_Lines_per_Season.png` — total number of lines in each season  

