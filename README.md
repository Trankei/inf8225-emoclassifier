# inf8225-emoclassifier

Since the models are to big to push into github you should run these codes to build the models :

1- code/text_classifier/textClassifierTraining.py : Build text_classifier.bin
2- code/image_classifier/ImageClassifierTraining.ipynb : Build image_classifier_71precision.pkl
3- code/image_classifier/Image_to_Vect.ipynb : Build inception_feature_extractor.h5 and inception_feature_extractor.json 
4 - code/early_fusion/earlyFusionTraining.py : Build early_fusion_model.tar
5- code/joint_fusion/jointFusionTraining.py : Build joint_fusion_model.bin
6- code/late_fusion/LateFusion.ipynb : Build late_fusion_models.pkl

Also, the GloVe words vectors should be downloaed from this link : "http://nlp.stanford.edu/data/glove.twitter.27B.zip" into the word_vectors directory.

The dataset is already available in the "processed_data" folder. The data was built using the TrainTestSplitData_random.ipynb

