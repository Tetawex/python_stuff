def map_column(column, strategy):
    if strategy == 'Id':
        return column
    elif strategy == 'Boolean':
        return column.astype('str').map({'t': 1, 'f': 0}).astype('Int64')
    elif strategy == 'Percentage':
        return column.str.rstrip('%').astype('Float64').divide(100)
    elif strategy == 'Price':
        return column.str.replace(',', '').str.replace('$', '').astype('Float64')
    elif strategy == 'Integer':
        return column.astype('Int64')
    elif strategy == 'Float':
        return column.astype('Float64')
    else:
        return None
