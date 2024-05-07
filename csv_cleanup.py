import pandas as pd

FILENAME = 'historic_deals.csv'
FILEPATH = ''
XCOLUMNS = ['ReleaseDate', 'OriginalPrice', 'DiscountPCT']
YCOLUMN = 'DiscountPCT'
NUMBERSLIST = ['0','1','2','3','4','5','6','7','8','9']

def process_file(filename, x_columns, y_column, filepath=''):
    df = pd.read_csv(filepath + filename)
    df = df.dropna()
    
    #print(df.iloc[1:3, 3:])
    #print(df.shape)
    for index, row in df.iterrows():
        for col in x_columns:
            if isinstance(row[col], float):
                df = df.drop(index)
                break
        
        if (''.join([i for i in row[y_column] if i in NUMBERSLIST]) == ''):
            df = df.drop(index)
            #print("dropped", index)

    #print(df.shape)
    # This section stolen from Rachelle
    x_dataframe = df[x_columns].copy()

    # Clean up the date column
    x_dataframe.insert(0, 'Day', 0)
    x_dataframe.insert(0, 'Month', 0)
    x_dataframe.insert(0, 'Year', 0)

    for index, row in x_dataframe.iterrows():
        day, month, year = row['ReleaseDate'].split("/")
        x_dataframe.loc[index, 'Day'] = int(day)
        x_dataframe.loc[index, 'Month'] = int(month)
        x_dataframe.loc[index, 'Year'] = int(year)

        x_dataframe.loc[index, 'OriginalPrice'] = float(''.join([i for i in row['OriginalPrice'] if i in NUMBERSLIST])) / 100
        x_dataframe.loc[index, y_column] = float(''.join([i for i in row[y_column] if i in NUMBERSLIST]))
            

    # This date bit stolen from Rachelle as well

    x_dataframe['date'] = pd.to_datetime(x_dataframe[['Year', 'Month', "Day"]])
    earliest_date = x_dataframe['date'].min()
    x_dataframe['date_numeric'] = (x_dataframe['date'] - earliest_date)

    # Drop unnecessary rows - date_numeric is all we need now.
    x_dataframe = x_dataframe.drop(['Day', 'Month', 'Year', 'ReleaseDate'], axis=1)

    y_dataframe = df[[y_column]].copy()
    for index, row in y_dataframe.iterrows():

        y_dataframe.loc[index, y_column] = float(''.join([i for i in row[y_column] if i in NUMBERSLIST])) / 100
    
    return x_dataframe, y_dataframe
    '''with open (filepath + filename, 'r') as source_file:
        filedata = ''
        max_price = 0
        max_date = MAX_DATE
        for line in source_file:
            split_line = line.split(',')
            if (split_line[3][-1] == '%'):
                numbers = int(split_line[4][3:])

    
    

    with open (newfilepath + newfilename, 'w') as destination_file:
        destination_file.write(filedata)
    
    return pd.read_csv(newfilepath + newfilename)
    '''

if __name__ == '__main__':
    process_file(FILENAME, XCOLUMNS, YCOLUMN, FILEPATH)