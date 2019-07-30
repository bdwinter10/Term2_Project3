import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv('data/disaster_categories.csv')
    return messages, categories

def clean_data(messages,categories):
    df = messages.merge(categories,how='left',on='id')
    splitted = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row =splitted.iloc[0] #grab the first row for the header
    category_colnames = [sub[ : -2] for sub in row] 
    for name,colname in zip(category_colnames,splitted.columns):
        df[name]=splitted[colname]
    for name in category_colnames:
        # set each value to be the last character of the string
        df[name] = df[name].str[-1:]

        # convert column from string to numeric
        df[name] = df[name].astype(int)
    df.drop(columns=['categories'],inplace=True)
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('cleaned_df', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()