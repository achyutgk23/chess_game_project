def text_in_plot(ax):
    '''
    INPUT:
    ax - axes of a plot
    
    OUTPUT:
    Annotates the text(height of the bar) at the top of the bar
    '''
    for p in ax.patches:
        ax.annotate('%.2f' % p.get_height(), 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   xytext=(3,6), ha='center', va='bottom',
                   textcoords="offset points",
                   fontsize=10, rotation=55)

def title(title):
    '''
    INPUT:
    title - Takes title of a plot as input
    
    OUTPUT:
    Outputs a title for a graph
    '''
    if True:
        plt.title(title, y=1.15, loc='left', fontsize=15.5, ha='left',
                 color='Blue')

def model_selection(classifier, X_train, y_train):
    '''
    INPUT:
    classifier - From the list of classifiers considering one classifier at a time
                 to evaluate its performance on validation sets(created by training set)
    X_train - Pandas Dataframe with predictors/variables
    y_train - Pandas Series with corresponding labels
    
    OUTPUT:
    1. A classifier from the list of classifiers predicts on 
       different validation sets and the predictions are stored in preds variable
    2. Prints scores of different evaluation metrics like
        a. Accuracy
        b. Precision
        c. Recall
        and prints the label from y_train which the classifier did not 
        predict
    '''
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from sklearn.model_selection import cross_val_predict
    
    classifier.fit(X_train, y_train)
    preds = cross_val_predict(classifier, X_train, y_train, cv=3,
                          n_jobs=-1)
    print('accuracy:', accuracy_score(y_train, preds))
    print('precision:', precision_score(y_train, preds, average='weighted'))
    print('recall:', recall_score(y_train, preds, average='weighted'))
    print(set(y_train) - set(preds))

def show_scores(y_train, preds):
    '''
    INPUT:
    y_train - A pandas series with target variable meant for
              training a classifier
    preds - Predictions made by a classifier on training set (X_train)
    
    OUTPUT:
    Outputs scores of different evaluation metrics for a classifier like
        1. Accuracy
        2. Precision
        3. Recall
        4. Confusion matrix
    '''
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import precision_score, recall_score
    
    print('accuracy:', accuracy_score(y_train, preds))
    print('precision:', precision_score(y_train, preds, average='weighted'))
    print('recall:', recall_score(y_train, preds, average='weighted'))
    print('confusion_matrix:')
    print(confusion_matrix(y_train, preds))










    
    print(set(y_train) - set(preds))
