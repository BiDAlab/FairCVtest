
# FairCVtest: Testbed for Fair Automatic Recruitment and Multimodal Bias Analysis


![](https://github.com/BiDAlab/FairCVtest/blob/master/Figures/CV.PNG)
**Figure 1. Information blocks in a resume and personal attributes that can be derived from each one. The number of crosses represent the level of sensitive information (+++ =  high, ++ = medium, + = low).**


We present a new experimental framework aimed to study how multimodal machine learning is influenced by biases present in the training  datasets. The framework is designed as a fictitious automated recruitment system, which takes a feature vector with data obtained from a resume as input to predict a score  within the interval [0, 1]. We have generated 24,000 synthetic resume profiles including 12 features obtained from 5 information blocks and 2 demographic  attributes  (gender  and  ethnicity), and a feature embedding with 20 features extracted from a face photograph. The 5 information blocks are: education attainment (generated from US Census Bureau 2018 Education Attainment data, without gender or ethnicity distinction), availability, previous experience, the existence of a recommendation letter, and language proficiency in a set of 8 different and common languages. We refer to the information from these 5 blocks as candidates competencies. Each profile has been associated according to the gender  and  ethnicity  attributes  with  an  identity  of  the  DiveFace  database [2], from which we get the face photographs.

Each resume is scored using a linear combination of the candidates competencies, adding a slight Gaussian noise to introduce a small degree of variability. Since we're not taking into account gender or ethnicity information during the score generation, these become agnostic to this information and should be equally distributed among different demographic groups. Thus, we refer to this target function as Unbiased scores, from which we define two target functions that include two types of bias, Gender bias and Ethnicity bias. Biased scores are generated by applying a penalty factor to certain individuals belonging to a particular demographic group.

We use the pretrained model ResNet-50 to extract feature embeddings from the face photograps. ResNet-50’s last convolutional layer outputs embeddings with 2048 features, so we added a fully connected layer to perform a bottleneck that compresses these embeddings to just 20 features (maintaining competitive face  recognition  performances). Despite being trained exclusively for the task of face recognition, the embeddings extracted by ResNet-50 contain enough information to infer gender and ethinicity, as this information is part of the face attributes. For this reason, we also extract feature embeddings applying to the pretrained model the method proposed in [1] to remove sensitive information, and so obtaining gender/ethnicity agnostic feature embeddings.



![](https://github.com/BiDAlab/FairCVtest/blob/master/Figures/figure_learning_network.PNG)
**Figure 2. Multimodal learning architecture composed by a Convolutional Neural Network (ResNet-50) and a fully connectednetwork used to fuse the features from different domains (image and structured data).  Note that some features are includedor removed from the learning architecture depending of the scenario under evaluation.**


# FairCVtdb

This framework present the gender and ethinicty cases as two separate but analogous experiments, maintaining a similar structure in both cases. Of the 24,000 synthetic profiles generated for each experiment, we retain the 80% (i. e. 19,200 CVs) as training set, and leave the remaining 20% (i. e. 4,800 CVs) as validation set. Both splits are equally distributed among the demographic attribute of the experiment. You can donwload the **gender profiles** here [[Training set](https://github.com/BiDAlab/FairCVtest/blob/master/data/Profiles_train.npy)] [[Validation set](https://github.com/BiDAlab/FairCVtest/blob/master/data/Profiles_test.npy)], and the **ethnicity profiles** here [[Training set](https://github.com/BiDAlab/FairCVtest/blob/master/data/Profiles_train_et.npy)] [[Validation set](https://github.com/BiDAlab/FairCVtest/blob/master/data/Profiles_test_et.npy)]. The following example illustrates how to load the information in python:
```python

import numpy as np

dict_profiles = np.load(profiles_path, allow_pickle = True).item()
profiles_feat = dict_profiles['profiles']
biased_labels = dict_profiles['biasedLabels']
blind_labels = dict_profiles['blindLabels']
image_list = dict_profiles['image_list']

```

The **profiles_feat** variable is a numpy array, where each row stores a different resume. You can access each of the i-th profile's attributes as follows:

```python

ethnicity = profiles_feat[i,0] # 0 = G1, 1 = G2, 3 = G3
gender = profiles_feat[i,1] # 0 = Male, 1 = Female
educ_attainment = profiles_feat[i,2] # Discrete variable [0 - 5]
prev_experience = profiles_feat[i,3] # Continuous variable [0 - 4]
recommendation = profiles_feat[i,4] # Binary variable
availability = profiles_feat[i,5] # Discrete variable [1 - 5]
language_prof = profiles_feat[i,6:14] # Discrete variables [0 - 3]
face_embedding = profiles_feat[i,14:34]
agnostic_face_embedding = profiles_feat[i,34:]

```

You can also download the results for each of the Scenarios presented in [3], which are defined as follows:

   - **Scenario 1** was trained with the candidates competencies, the demographic attributes and the Unbiased scores. You can download the results here [[Gender results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_blind_feat_gender.npy)] [[Ethnicity results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_blind_feat_ethnicity.npy)].
   
   - **Scenario 2** was trained with the candidates competencies, the demographic attributes and the Gender/Ethinicty Biased scores. You can download the results here [[Gender results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_feat_gender.npy)] [[Ethnicity results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_feat_ethnicity.npy)].
   
   - **Scenario 3** was trained with the candidates competencies and the Gender/Ethnicity Biased scores. You can download the results here [[Gender results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_gender.npy)] [[Ethnicity results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_ethnicity.npy)].
   
   - **Scenario 4** was trained with the candidates competencies, the face embeddings and the Gender/Ethnicity Biased scores. You can download the results here [[Gender results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_facial_gender.npy)] [[Ethnicity results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_facial_ethnicity.npy)].
   
   - **Scenario 4 (Agnostic)** was trained with the candidates competencies, the agnostic face embeddings and the Gender/Ethnicity Biased Scores. You can download the results here [[Gender results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_facial_ag_gender.npy)] [[Ethnicity results](https://github.com/BiDAlab/FairCVtest/blob/master/data/predictions_biased_facial_ag_ethnicity.npy)].
  
The following example illustrates how to load the results in python:

```python

import numpy as np

dict_results = np.load(results_path, allow_pickle = True).item()
val_pred = dict_results['predictions']
val_loss = dict_results['history_loss']

```

# License

Any entity using this dataset agrees to the following conditions:

THIS DATASET IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# References

For further information on the benchmark and on different applications where it has been used, we refer the reader to (all these articles are publicly available in the [publications](http://atvs.ii.uam.es/atvs/listpublications.do) section of the BiDA group webpage).

[1] A. Peña, I. Serna, A.   Morales, and   J.   Fierrez, “Understanding Biases in Multimodal AI: Case Study in Automated Recruitment,” Proc. of IEEE CVPR Workshop on Fair, Data Efficient and Trusted Computer Vision, Washington, Seattle, USA, 2020.

[2] A.   Morales,   J.   Fierrez,   and   R.   Vera-Rodriguez, “Sensitivenets: Learning   Agnostic Representations with  Application  to  Face Recognition,” arXiv:1902.00334, 2019.

Please remember to reference articles [1,2] on any work made public, whatever the form, based directly or indirectly on any part of the FairCVtest benchmark.


# Contact:

For more information contact Aythami Morales, associate professor UAM at aythami.morales@uam.es
