import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Copy the data to avoid altering the original dataframe
        X = X.copy()
        
        # 1. Family Size
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

        # 2. IsAlone: a binary feature indicating if a passenger is alone
        X['IsAlone'] = np.where(X['FamilySize'] == 1, 1, 0)
        
        # 3. Title Extraction from the Name
        X['Title'] = X['Name'].apply(self.extract_title)
        
        # 4. CabinDeck extraction from Cabin feature (use only first letter as deck)
        X['CabinDeck'] = X['Cabin'].apply(self.extract_deck)
        
        # 5. AgeGroup (binning Age into categories)
        X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 12, 18, 35, 60, np.inf], labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])

        # 6. Fare per person (based on family size)
        X['FarePerPerson'] = X['Fare'] / X['FamilySize']
        
        # Drop irrelevant columns for this stage
        X.drop(columns=['Ticket', 'Name', 'Cabin'], inplace=True)
        
        return X

    def extract_title(self, name):
        # Extract title from the name using simple string manipulation
        title = name.split(',')[1].split('.')[0].strip()
        # Normalize some rare titles into a few common ones
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master', 
            'Rev': 'Other', 'Dr': 'Other', 'Col': 'Other', 'Major': 'Other', 
            'Mlle': 'Miss', 'Countess': 'Royalty', 'Ms': 'Miss', 'Lady': 'Royalty',
            'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Sir': 'Royalty', 'Mme': 'Mrs', 
            'Capt': 'Other'
        }
        return title_mapping.get(title, 'Other')

    def extract_deck(self, cabin):
        # Extract the first letter from the Cabin string (indicating the deck)
        if pd.isnull(cabin):
            return 'Unknown'
        else:
            return cabin[0]



class ExtendedTitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Family Size
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

        # 2. IsAlone: a binary feature indicating if a passenger is alone
        X['IsAlone'] = np.where(X['FamilySize'] == 1, 1, 0)

        # 3. Title Extraction from the Name
        X['Title'] = X['Name'].apply(self.extract_title)

        # 4. CabinDeck extraction from Cabin feature (use only first letter as deck)
        X['CabinDeck'] = X['Cabin'].apply(self.extract_deck)

        # 5. AgeGroup (binning Age into categories)
        X['AgeGroup'] = pd.cut(X['Age'], bins=[0, 12, 18, 35, 60, np.inf], labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Senior'])

        # 6. Fare per person (based on family size)
        X['FarePerPerson'] = X['Fare'] / X['FamilySize']

        # 7. Fare Binning
        X['FareBin'] = pd.qcut(X['Fare'], 4, labels=['Low', 'Medium', 'High', 'Premium'])

        # 8. Ticket Frequency (counting occurrences of the same ticket)
        X['TicketFrequency'] = X.groupby('Ticket')['Ticket'].transform('count')

        # 9. Surname extraction and grouping
        X['Surname'] = X['Name'].apply(lambda name: name.split(',')[0])
        X['SurnameGroup'] = X.groupby('Surname')['Surname'].transform('count')

        # 10. Child/Mother Indicator
        X['IsChild'] = np.where(X['Age'] < 16, 1, 0)
        X['IsMother'] = np.where((X['Sex'] == 'female') & (X['Parch'] > 0) & (X['SibSp'] == 0), 1, 0)

        # 11. Interaction Feature: Sex * Pclass
        X['Sex_Pclass'] = X['Sex'] + '_' + X['Pclass'].astype(str)

        # 12. Cabin availability (whether Cabin was recorded or not)
        X['HasCabin'] = np.where(X['Cabin'].isnull(), 0, 1)

        # Drop irrelevant columns
        X.drop(columns=['Ticket', 'Name', 'Cabin'], inplace=True)
        
        return X

    def extract_title(self, name):
        # Extract title from the name using simple string manipulation
        title = name.split(',')[1].split('.')[0].strip()
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master', 
            'Rev': 'Other', 'Dr': 'Other', 'Col': 'Other', 'Major': 'Other', 
            'Mlle': 'Miss', 'Countess': 'Royalty', 'Ms': 'Miss', 'Lady': 'Royalty',
            'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Sir': 'Royalty', 'Mme': 'Mrs', 
            'Capt': 'Other'
        }
        return title_mapping.get(title, 'Other')

    def extract_deck(self, cabin):
        # Extract the first letter from the Cabin string (indicating the deck)
        if pd.isnull(cabin):
            return 'Unknown'
        else:
            return cabin[0]

