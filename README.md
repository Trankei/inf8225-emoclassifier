# inf8225-emoclassifier

Since the models are to big to push into github you should run these codes to build the models :

## Required libraries:
* pytorch
* bcolz
* scikitlearn
* numpy
* fastText

## Models:

### Late Fusion
code/late_fusion/LateFusion.ipynb : Build late_fusion_models.pkl

#### Text classifier
code/text_classifier/textClassifierTraining.py : Build text_classifier.bin

#### Image classifier
code/image_classifier/ImageClassifierTraining.ipynb : Build image_classifier_71precision.pkl

### Early Fusion
code/early_fusion/earlyFusionTraining.py : Build early_fusion_model.tar

### Joint Fusion
 code/joint_fusion/jointFusionTraining.py : Build joint_fusion_model.bin
 
### Image to vector
code/image_classifier/Image_to_Vect.ipynb : Build inception_feature_extractor.h5 and inception_feature_extractor.json 

Also, the GloVe words vectors should be downloaed from this link : http://nlp.stanford.edu/data/glove.twitter.27B.zip into the word_vectors directory.

The dataset is already available in the "processed_data" folder. The data was built using the TrainTestSplitData_random.ipynb

