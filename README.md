# inf8225-emoclassifier

Since the models are to big to push into github you should run these codes to build the models :

## Required libraries:
* pytorch
* bcolz
* scikitlearn
* numpy
* fastText

## Notes:
> The GloVe words vectors should be downloaded from this link : http://nlp.stanford.edu/data/glove.twitter.27B.zip. The `glove.twitter.27B.200d.txt` file extracted from the downloaded zip should placed into a `word_vectors` folder in the root directory.

> The dataset is already available in the `processed_data` folder. The data was built using the `TrainTestSplitData_random.ipynb`.

## Models:
### Late Fusion
To generate Late Fusion model `late_fusion_models.pkl`, run
> `code/late_fusion/LateFusion.ipynb`

##### Text classifier
To train text classifier `text_classifier.bin`, run
> `code/text_classifier/textClassifierTraining.py`

##### Image classifier
To train image classifier `image_classifier_71precision.pkl`, run 
> `code/image_classifier/ImageClassifierTraining.ipynb`

### Image to vector
To generate image vectors `inception_feature_extractor.h5` and `inception_feature_extractor.json`, run
> `code/image_classifier/Image_to_Vect.ipynb`

### Early Fusion
To train early fusion model `early_fusion_model.bin`, run 
> `code/early_fusion/earlyFusionTraining.py`

### Joint Fusion
To train joint fusion model `joint_fusion_model.bin`, run
> `code/joint_fusion/jointFusionTraining.py`

