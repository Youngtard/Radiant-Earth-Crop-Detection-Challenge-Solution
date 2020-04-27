### Approach
I used two different approaches. The first approach involved training with 3 set of features:
* Image pixel values 
* About 10 vegetation/spectral indices (e.g. NDVI, AVI etc.), and their relevant statistics 
* Spatial features (e.g area of farm etc.).  
The second approach involved training with only pixel values and their relevant statistics.  
My solution is an ensemble weighted average of the two approaches.

### Modelling
The two approaches each went through the same modelling process by using a CatboostClassifier (without class_weights), another CatboostClassifier (with class_weights to take care of class imbalance), and a LinearDiscriminant algorithm (known in sklearn as LinearDiscriminantAnalysis - LDA ). LDA is a weak learner, so in order to improve it's performance, I bagged (ensemble) it using sklearn's BaggingClassifier. The weighted Catboost and bagged LDA added some diversity to the modelling due to the highly imbalanced dataset. Using just the single Catboost with no class_weights, I was having about 1.18 on the Public Leaderboard. By adding the two other algorithms subsequently, my score gradually improved to about 1.14 on the Public Leaderboard.