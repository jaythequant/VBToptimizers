from sklearn.model_selection import KFold


def vbt_cv_scikit_constructor(df, cv):
    """
    Cross validator using vectorbt in combination with SKlearn CV model
    :param df: Dataframe object for constructor to split
    :param cv: SKlearn cross_validator (e.g., TimeSeriesSplit)
    :return: Tuple of list of dataframes split in accordance with cv method used
    """
    (train_df, train_idx), (test_df, test_idx) = df.vbt.split(cv)

    train_folds = []
    test_folds = []

    for i in train_df.columns.levels[0]:
        df = train_df.loc[:, i].dropna()
        df.index = train_idx[i]
        train_folds.append(df)

    for i in test_df.columns.levels[0]:
        df = test_df.loc[:, i].dropna()
        df.index = test_idx[i]
        test_folds.append(df)

    return train_folds, test_folds


def vbt_cv_kfold_constructor(df, n_splits=5, shuffle=False, random_state=None):
    """
    Cross validator using vectorbt in combination with SKlearn KFold CV
    :param df: Dataframe object for constructor to split
    :param cv: Disables shuffling of data. Defaults to false as most VBT ML data is time series
    :param random_state: Seeds shuffle effect for replication purposes. Not neccesary is shuffle=False
    :return: Tuple of list of dataframes split as n folds of training and testing data
    """
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    (train_df, train_idx), (test_df, test_idx) = df.vbt.split(cv)

    train_folds = []
    test_folds = []

    for i in train_df.columns.levels[0]:
        df = train_df.loc[:, i].dropna() # If row length is odd, there will be 1 nan
        df.index = train_idx[i]
        train_folds.append(df)

    for i in test_df.columns.levels[0]:
        df = test_df.loc[:, i].dropna() # If row length is odd, there will be 1 nan
        df.index = test_idx[i]
        test_folds.append(df)

    return train_folds, test_folds


def vbt_cv_sliding_constructor(df, n_splits, set_lens=(), min_len=1):
    """
    Cross validator using vectorbt implementation of sliding window CV
    :param df: Dataframe object for constructor to split
    :param n_splits: Integer input determining number of splits to make
    :params set_lens: Int or float Optional input used to further decompose data
        Set_lens may take the form (x,) or (x,y)
        If input set_lens is form (x,) => function returns two sets per n_splits
            with the first returned set as training data of x length (int or pct)
            and the second returned set as testing data of x length (if int) or
            (1 - x)% length if float was inputted
        set_len form of (x,y) returns a validation set in addition to train and test
    :params min_len: Integer minimum valid set length. 
    :returns: List of dataframes in slider CV intervals
    """
    if not set_lens:
        """Returns single list of dataframes with no explicit test data"""
        try: 
            split_sets, split_idx = df.vbt.range_split(n=n_splits, min_len=min_len, set_lens=set_lens)
        except ValueError as e:
            print("ValueError:", e)

        split_folds = []

        for i in split_sets.columns.levels[0]:
            df = split_sets.loc[:, i].dropna()
            df.index = split_idx[i]
            split_folds.append(df)

        return split_folds

    elif len(set_lens) == 1:
        """Splits each n_split into training and testing dataframes returning tuple object"""
        try:
            (train_df, train_idx), (test_df, test_idx) = df.vbt.range_split(n=n_splits, min_len=min_len, set_lens=set_lens)
        except ValueError as e:
            print("ValueError:", e)

        train_folds = []
        test_folds = []

        for i in train_df.columns.levels[0]:
            df = train_df.loc[:, i].dropna()
            df.index = train_idx[i]
            train_folds.append(df)

        for i in test_df.columns.levels[0]:
            df = test_df.loc[:, i]
            df.index = test_idx[i]
            test_folds.append(df)
        
        return train_folds, test_folds


    elif len(set_lens) == 2:
        """
        Split each n_split into training, validation, and testing dataframes 
        returning tuple object
        """
        try:
            (train_df, train_idx), (validate_df, validate_idx), (test_df, test_idx) = df.vbt.range_split(
                n=n_splits, min_len=min_len, set_lens=set_lens
            )
        except ValueError as e:
            print("ValueError:", e)

    train_folds = []
    validate_folds = []
    test_folds = []

    for i in train_df.columns.levels[0]:
        df = train_df.loc[:, i].dropna()
        df.index = train_idx[i]
        train_folds.append(df)

    for i in validate_df.columns.levels[0]:
        df = validate_df.loc[:, i]
        df.index = validate_idx[i]
        test_folds.append(df)

    for i in test_df.columns.levels[0]:
        df = test_df.loc[:, i]
        df.index = test_idx[i]
        test_folds.append(df)

    return train_folds, validate_folds, test_folds
