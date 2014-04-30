"""
It is a lot more fun if this data is personal to you. 
Go to the training set data (metro_100_train) and clear out all the data. Keep the rows.
Now, find 10 cities where you can say whether you would definitely want to live there (at least 3 cities)/definitely not want to live there (at least 3 cities). Try to find at least 10 overall. 
If the city is somewhere you want to live, put a 1 in Column A.
If the city is somewhere you do not live, put a 0 in Column A.
Fill out the rest of the columnn information for each city (copy/paste works).
Now when you run the script, the model will be trained on your preferences!
"""

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('metro_100_train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('metro_100_test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(train, target)
    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

    savetxt('submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', 
            header='Id,PredictedProbability', comments = '')

if __name__=="__main__":
    main()
