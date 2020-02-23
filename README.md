# Sourcing Sound
## Audio classification



### Project Description and Motivation
- Identify sound from many possible sources (speech, animals, vehicles, ambient noise, etc.)
- Wide variety of applications for audio recognition:
    - Manufacturing quality control
    - Domestic animal monitoring
    - Wildlife monitoring
    - Security event detection
    - Music production
- Initially develop pipeline for single class
- Expand to multi-class



## Datasets
- [Freesound Dataset](https://annotator.freesound.org/fsd/) - manually labeled by open source
    - 4969 clips
    - 70 examples per class
    - Total duration: ~ 10 hours
- Portion of [Yahoo Flickr Creative Commons (YFCC) 100M dataset](https://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/) - 
    - 19,815 clips
    - 300 examples per class
    - Total duration: ~80 hours

## Processing
- Explored various metrics: Spectral centroid, spectral bandwidth, Spectral rolloff, zero crossing rate, MFCCs, Mel Spectrograms
- Because this project is not soley focused on human speech, Mel spectrograms are likely more appropriate. However, MFCCS were also explored, as well as raw audio, however they did not achieve the best results and generally required more computational resources.

Mel spectrograms give us an 'image' of a sound. On the Mel scale, distance between frequencies better relate to human hearing; the difference between 100 - 200hz is extremely noticeable to humans, but 10,000-10,100hz is barely audible. The Mel scale takes this into account by logarithmic scaling of frequencies into multiple filter banks.

![Mel Spectrogram of a purr](img/purr.png)
![unprocessed frequency power](img/mel_winoriginal.png)
![log frequency power normalized](img/mel_winnorm.png)

## Baseline Model

Initially, the class labels were reduced to a simple recognition problem: does a clip contain the sound "purr" or not?


A gradient boosting classifier was built, and trained on 3/4 of the training files, and when evaluated with the hold out set, achieved an  an accuracy of 98% and recall and precision of 0%. Even after grid searching over parameters including tree-depth, learning rate, number of estimators, the results 

The extremely low true positive rate was likely due to the model mostly being provided with "non-purr" samples, which allowed it to increase it's accuracy by simply predicting a sample was not a purr.

Other issues were probably coming from noisy data/labels, and differing sample durations.

### Initial Improvements
### Overlapping windows

To give the classifier more contextual information on the same clip, implement rolling, overlapping windows on the Mel spectrograms of the sound.

![](img/mel_win215.png)
![](img/mel_win268.png)
![](img/mel_win321.png)
![](img/mel_win374.png)


<b>Minor success:</b> The precision increased to 52%. Recall increased to 6.6%. Accuracy dropped to 51.7%.


<b>Checkpoint, next steps:</b> Gradient boosting classifiers are a weak approach to this problem. The multiple frequency bins for every time frame are flattened end to end, thus destroying the shape of the spectrograms. Using modern image classification techniques, teaching a model to recognize sounds within a noisy 2D context may yield better results.


# Building a CNN Model

## Feature extraction improvements
- 



## RNN Model