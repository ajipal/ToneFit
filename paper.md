## III. Methodology

---

### 1. Data Acquisition

#### Data Source / Participants

The dataset used in this study is the Deep Armocromia dataset introduced by Stacchio et al. (2024), publicly available from the VRAI Lab GitHub repository. Rather than involving human participants directly, this study uses a curated image dataset of human faces labeled by domain experts. The dataset comprises 4,920 facial images labeled across 4 personal color seasons — Spring, Summer, Autumn, and Winter — and further subdivided into 12 sub-types by certified Armocromia professionals trained in the Flow Theory methodology. The dataset was selected because it is the only publicly available benchmark specifically designed for seasonal color classification.

| Dataset | Images | Labels | Validation |
|---|---|---|---|
| Deep Armocromia | 4,920 | 4 seasons (12 sub-types) | Certified Armocromia analysts, ECCV 2024 |

**Table 1.** Summary of the Deep Armocromia dataset used in this study.

#### Steps, Techniques, and Tools in Data Collection

The Deep Armocromia dataset was assembled by Stacchio et al. (2024) from two distinct sources. The first source is the CelebA dataset, a large-scale face image collection from which 2,981 samples were selected. The second source is the Pivotal Armocromia Set, a curated collection of 1,939 images of public figures with publicly documented Armocromia season diagnoses. Season labels for all images were assigned by certified Armocromia analysts following the Flow Theory color methodology, ensuring professional-grade ground truth annotations. The dataset is provided with a predefined train/test split and per-image segmentation masks, and was accessed directly from the VRAI Lab GitHub repository without modification to its structure or labels. No additional data collection was performed in this study.

---

### 2. Data Pre-processing

#### Data Cleaning and Preparation

The dataset underwent a structured cleaning and preparation pipeline to ensure consistency and model readiness. Duplicate images were first identified and removed using the 16-bit Average Perceptual Hash (pHash) algorithm, following the same deduplication methodology employed in the original study. All remaining images were resized to 224×224 pixels to meet the fixed input requirements of the Vision Transformer backbone. Face segmentation was applied using the Facer toolbox masks provided in RGB-M format within the dataset, isolating key facial regions — hair, skin, nose, eyebrows, eyes, and mouth — from background elements. Images with failed face detection or insufficient face size were excluded to maintain consistent input quality across all training and test samples.

#### Feature Extraction

Feature extraction is handled differently for each model based on their respective input requirements. FaRL performs implicit feature extraction entirely within its backbone, while the SVM requires explicit handcrafted color features derived from each image.

**FaRL — Implicit Feature Extraction.**
FaRL does not require a manual feature extraction step. The 224×224 face image is divided into non-overlapping 16×16 pixel patches, producing 196 patch tokens. Each token is linearly projected into a 512-dimensional embedding space and processed through 12 Transformer blocks using multi-head self-attention. The output [CLS] token represents a 512-dimensional learned feature embedding of the entire face, encoding color, texture, contrast, and facial structure simultaneously. Because FaRL was pretrained on 50 million face-text pairs from the LAION-Face dataset using contrastive learning, its embeddings encode rich face-specific representations — including skin tone and undertone — without any manual feature definition. The backbone is used in frozen mode throughout training.

**SVM — Explicit Feature Extraction.**
For the SVM, a 7-dimensional feature vector is extracted from each face image using the CIELab and HSV color spaces alongside the Individual Typology Angle (ITA) score. The CIELab color space was selected because it is perceptually uniform, making it the most appropriate space for capturing the subtle undertone differences that define Armocromia seasons. Six CIELab statistics are computed per image: the mean and standard deviation of each axis (L*_mean, a*_mean, b*_mean, L*_std, a*_std, b*_std), where a* encodes warm/cool undertone and b* encodes golden/cool direction. The HSV color space provides complementary information: H_mean captures dominant hue angle, S_mean captures color vibrancy, and V_mean captures overall brightness.

| Feature | Axis | Range | Armocromia Interpretation |
|---|---|---|---|
| L*_mean | Lightness | 0–100 | High L* → lighter skin (Summer/Winter); Low L* → darker skin (Autumn) |
| a*_mean | Red–Green | −80 to +80 | Positive → warm undertone (Spring/Autumn); Negative → cool undertone (Summer/Winter) |
| b*_mean | Yellow–Blue | −80 to +80 | Positive → golden/warm (Spring/Autumn); Negative → cool/ashy (Summer/Winter) |
| L*_std | Lightness variation | ≥ 0 | High std → high facial contrast (Winter); Low std → muted/soft features (Summer) |
| a*_std | Red–Green variation | ≥ 0 | Captures variation in warm/cool tones across face |
| b*_std | Yellow–Blue variation | ≥ 0 | Captures variation in warm/cool tones across face |

**Table 2.** CIELab features extracted from each face image and their interpretation for Armocromia season classification.

The Individual Typology Angle (ITA) is a validated scalar metric that combines CIELab lightness and undertone into a single discriminative score, computed as:

> ITA = arctan((L\*_mean − 50) / b\*_mean) × (180/π)

Higher ITA values indicate lighter, cool-toned skin associated with Summer and Winter, while lower ITA values indicate darker, warm-toned skin associated with Autumn and Spring. Table 3 presents the ITA interpretation ranges.

| ITA Range | Skin Type | Associated Seasons | Undertone |
|---|---|---|---|
| ITA > 55° | Very light | Summer, Winter (Cool) | Cool, neutral |
| 28° < ITA ≤ 55° | Light | Spring (Warm) | Warm, golden |
| 10° < ITA ≤ 28° | Intermediate | Spring / Summer | Neutral |
| −30° < ITA ≤ 10° | Tan / Olive | Autumn (Warm) | Warm, earthy |
| ITA ≤ −30° | Dark | Deep Autumn, Deep Winter | Deep, warm or cool |

**Table 3.** ITA score ranges and their corresponding Armocromia season associations. Adapted from Chardon et al. (1991) and Stacchio et al. (2024).

#### Data Transformation

For FaRL, no data augmentation is applied at any stage. Because the backbone is frozen, all images are passed through the ViT exactly once before training begins and the resulting 512-dimensional embeddings are saved to disk (feature caching). Applying augmentation would require re-running the full backbone for every augmented variant, defeating the purpose of caching. All images are resized to 256×256, center-cropped to 224×224, and normalized using CLIP-specific statistics (mean=[0.48145, 0.45783, 0.40821], std=[0.26863, 0.26130, 0.27577]) — the distribution the CLIP ViT-B/16 backbone was pretrained on — rather than ImageNet statistics.

For the SVM, all 7 extracted features are normalized to the [0, 1] range using MinMaxScaler from scikit-learn, fitted on the training set only and applied to the test set to prevent data leakage. Normalization is required because the SVM's RBF kernel is sensitive to feature scale differences across the extracted features.

---

### 3. Feature Engineering

#### Feature Encoding

For FaRL, labels are encoded using a custom dataset class (FaRLDataset) that walks the season/sub-type folder hierarchy and returns both a season index and a sub-type index per image. Season labels are assigned alphabetically: autumn=0, spring=1, summer=2, winter=3. Sub-type labels are also encoded as integers across all 12 classes in alphabetical order (autumn_deep=0, autumn_soft=1, autumn_warm=2, spring_bright=3, …, winter_deep=11). Both indices are produced per image to support the joint two-head training objective.

For the SVM, season labels are encoded as integer class indices in the same alphabetical order, ensuring consistent label definitions across both models.

#### Feature Selection

For FaRL, no manual feature selection is performed. The frozen backbone extracts a 512-dimensional embedding that implicitly encodes all relevant facial information, and all 512 dimensions are passed to the classifier head.

For the SVM, feature selection was guided by domain knowledge of the Armocromia color system. From the initial 10 possible features (6 CIELab statistics, 3 HSV statistics, 1 ITA score), the 7 most discriminative were selected: L*_mean, a*_mean, b*_mean, ITA, H_mean, S_mean, and V_mean. The standard deviation features (L*_std, a*_std, b*_std) were excluded following preliminary analysis, as their contribution to class separability was marginal compared to the mean and ITA features. The ITA score and a*_mean are the most discriminative features, as they directly encode the warm/cool undertone axis that defines the primary division between seasons. Table 4 summarizes the selected features and their roles.

| Feature | Source | Type | Discriminative Role |
|---|---|---|---|
| L*_mean | CIELab | Continuous | Lightness — separates deep (Autumn) from light (Summer/Winter) seasons |
| a*_mean | CIELab | Continuous | Warm/cool undertone — primary axis for Spring/Autumn vs Summer/Winter |
| b*_mean | CIELab | Continuous | Golden/warm vs cool/ashy tone — reinforces a* axis separation |
| ITA | Derived | Continuous | Single undertone+lightness score — highest discriminative power warm vs cool |
| H_mean | HSV | Continuous | Dominant hue angle — captures overall color warmth of skin tone |
| S_mean | HSV | Continuous | Color vibrancy — separates muted seasons (Summer) from vivid ones (Spring) |
| V_mean | HSV | Continuous | Brightness — complements L* lightness from CIELab space |

**Table 4.** Selected features for the SVM classifier and their discriminative roles in Armocromia season classification.

#### Data Split into Training and Test Sets

The Deep Armocromia dataset provides a predefined train/test partition established by Stacchio et al. (2024), which this study adopts without modification to ensure consistency. The training partition contains 4,008 images and the test partition contains 912 images, with season proportions preserved across both sets. FaRL loads images directly from the predefined split directories, while the SVM uses the corresponding rows from features.csv filtered by the split label. Due to preprocessing differences, the SVM was ultimately trained on 3,051 images and evaluated on 668 images — those for which all 7 color features were successfully extracted. This discrepancy is acknowledged in the Model Testing section.

| Season | Train | Test | Total | % of Total |
|---|---|---|---|---|
| Autumn | 1,046 | 259 | 1,305 | 26.5% |
| Spring | 978 | 203 | 1,181 | 24.0% |
| Summer | 943 | 186 | 1,129 | 22.9% |
| Winter | 1,041 | 264 | 1,305 | 26.5% |
| **Total** | **4,008** | **912** | **4,920** | **100%** |

**Table 5.** Predefined train/test partition of the Deep Armocromia dataset as established by Stacchio et al. (2024).

---

### 4. Model Development

#### Machine Learning Tools

This study develops and trains two models using separate machine learning frameworks. FaRL is implemented in PyTorch (v2.x), with the backbone loaded via the openai/CLIP library and trained on Google Colab with an L4 GPU. The SVM is implemented using scikit-learn (Python) and trained entirely on CPU. Both models share the same data loading, preprocessing, and label encoding pipelines to ensure consistency.

#### Classifiers, Attribute Selection Method, and Validation Algorithm

**Model A — FaRL (Deep Learning Classifier).**
FaRL (Face Representation Learning) is a Vision Transformer (ViT-Base/16) pretrained on 50 million face-text image pairs from the LAION-Face dataset using contrastive learning (Zheng et al., 2022). The pretrained weights correspond to the FaRL64 variant. The backbone is used as a frozen feature extractor. A flat two-head classifier is attached on top of the shared 512-dimensional backbone output:

```
shared FC(512→256) → ReLU → Dropout(0.5)
    ├── season_head:   FC(256→4)   [4-season classification]
    └── subtype_head:  FC(256→12)  [12-subtype classification]
```

Both heads receive the same shared intermediate representation and are trained simultaneously. Only the classifier heads are trained — the backbone parameters receive no gradient updates. The model is optimized using AdamW (lr=1e-3, weight_decay=1e-5) with a CosineAnnealingWarmRestarts scheduler (T_0=10, eta_min=1e-5). Training runs for 50 epochs with batch size 64. A joint CrossEntropyLoss is used: the season loss and sub-type loss are summed each step, with independent balanced class weights for each head. The best model checkpoint is selected by peak validation season accuracy.

| Configuration | Value |
|---|---|
| Backbone | FaRL (CLIP ViT-B/16, feature_dim=512) |
| Pretrained weights | FaRL-Base-Patch16-LAIONFace20M-ep64.pth (652 MB) |
| Feature dimension | 512 |
| Classifier head | Shared FC(512→256, ReLU, Dropout 0.5) → season_head FC(256→4) + subtype_head FC(256→12) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-5) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=10, eta_min=1e-5) |
| Epochs | 50 |
| Batch size | 64 |
| Loss function | Joint CrossEntropyLoss — CE(season) + CE(subtype), each with balanced class weights |
| Training hardware | Google Colab L4 GPU |

**Table 6.** FaRL model development configuration.

**Model B — SVM (Classical Computer Vision Classifier).**
The Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel serves as the classical computer vision baseline. SVM was selected because it is well-suited to compact handcrafted feature vectors, has demonstrated effectiveness for color-based facial classification tasks in prior literature (Lee et al., 2021), and requires no GPU — enabling a comparison of deep learning versus classical approaches across both performance and computational cost. Hyperparameter optimization is performed using GridSearchCV with 5-fold cross-validation on the training set. The search covers C=[0.1, 1, 10, 100] and gamma=['scale', 'auto'], and the best configuration (C=100, gamma='scale') is used to retrain the final model on the full training set.

| Configuration | Value |
|---|---|
| Model | Support Vector Machine (SVM) |
| Kernel | Radial Basis Function (RBF) |
| Input features | 7-dim CIELab/HSV/ITA vector (normalized) |
| Best hyperparameters | C=100, gamma=scale |
| Hyperparameter search | GridSearchCV — C=[0.1, 1, 10, 100], gamma=[scale, auto] |
| Cross-validation | 5-fold on training set |
| Class weight | balanced |
| Training time | 71.6 seconds |
| Training hardware | CPU (no GPU required) |
| Library | scikit-learn (Python) |

**Table 7.** SVM model development configuration.

#### Model Performance Metrics

Both models are evaluated using the same set of metrics to ensure a consistent and comparable assessment. The primary metric is the weighted F1-score, which balances precision and recall while accounting for class imbalance. Additional metrics include overall accuracy, weighted precision, weighted recall, Top-2 accuracy, per-class precision/recall/F1, a normalized confusion matrix, and Autumn recall tracked as a dedicated secondary metric. Table 8 defines each metric used in this study.

| Metric | Applies To | Description |
|---|---|---|
| Accuracy | FaRL, SVM | Proportion of correctly classified samples out of total test samples for the 4-class season task. |
| Weighted Precision | FaRL, SVM | Proportion of correct positive predictions per class, averaged with class-size weighting. |
| Weighted Recall | FaRL, SVM | Proportion of actual positives correctly identified per class, averaged with class-size weighting. |
| Weighted F1-Score | FaRL, SVM | Harmonic mean of weighted precision and recall. |
| Confusion Matrix | FaRL, SVM | N×N matrix showing predicted vs. actual class labels across all four seasons. |

**Table 8.** Performance metrics used for model evaluation in this study.

---

### 5. Model Testing

#### Classifiers and Test Set Application

Both trained models are applied to their respective held-out test sets following training completion, with no weight updates performed during testing. FaRL is evaluated on the full 912-image predefined test partition. The SVM is evaluated on 668 images — the subset of the test partition for which all 7 color features were successfully extracted. The same MinMaxScaler normalization parameters fitted during training are applied to the SVM test features. Season predictions are generated for all test images and compared against their ground-truth labels.

#### Model Performance Metrics — Test Results

Table 9 presents the test set performance for both models.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| FaRL | 0.564 | 0.567 | 0.564 | 0.555 |
| SVM | 0.457 | 0.479 | 0.457 | 0.451 |

**Table 9.** Test set performance of FaRL and SVM on the Deep Armocromia dataset.

[ INSERT FIGURE 2: FaRL vs SVM — Grouped bar chart comparing Accuracy and Weighted F1 ]

**Figure 2.** Grouped bar chart comparing accuracy and weighted F1-score for FaRL and SVM on the 4-season classification task.

FaRL achieved an overall accuracy of 0.564, weighted precision of 0.567, weighted recall of 0.564, and weighted F1-score of 0.555 on 912 test images. The SVM achieved an accuracy of 0.457, weighted precision of 0.479, weighted recall of 0.457, and weighted F1 of 0.451 on 668 test images, performing below FaRL on all metrics.

Table 10 presents per-class precision, recall, and F1-score for both models.

| Season | FaRL P | FaRL R | FaRL F1 | SVM P | SVM R | SVM F1 |
|---|---|---|---|---|---|---|
| Autumn | 0.583 | 0.436 | 0.499 | 0.450 | 0.531 | 0.487 |
| Spring | 0.500 | 0.434 | 0.464 | 0.393 | 0.293 | 0.335 |
| Summer | 0.510 | 0.554 | 0.531 | 0.448 | 0.453 | 0.450 |
| Winter | 0.618 | 0.795 | 0.695 | 0.524 | 0.574 | 0.548 |

**Table 10.** Per-class precision, recall, and F1-score for FaRL and SVM on their respective test sets.

[ INSERT FIGURE 3: Confusion Matrices — FaRL (left) vs SVM (right), side-by-side normalized heatmaps ]

**Figure 3.** Confusion matrices for FaRL (Acc=0.564, F1=0.555) and SVM (Acc=0.457, F1=0.451). Rows = actual labels; columns = predicted labels.

[ INSERT FIGURE 1: FaRL Flat Baseline Training Curves — Loss, Season Accuracy (4-class), Sub-Type Accuracy (12-class) over 50 epochs ]

**Figure 1.** FaRL training and validation curves over 50 epochs. Best epoch (epoch 6) is marked by the dashed green line. Validation season accuracy peaks at 0.564 before plateauing while training loss continues to decrease.

---

### 6. Model Evaluation

#### Comparison Between Model Development and Model Testing

During model development, FaRL's best validation accuracy was reached at epoch 6, after which validation performance plateaued while training loss continued to decrease across all 50 epochs. This gap indicates that the frozen backbone limits further generalization beyond its pretrained feature space. For the SVM, the best cross-validation F1-score during development was 0.434 with C=100, gamma='scale', which is closely reflected in the final test F1 of 0.451 — indicating that the SVM generalized consistently from training to testing without significant overfitting.

#### Significant Figures Contributing to Season Classification

Across both models, Winter is the most accurately classified season. FaRL achieves a Winter F1 of 0.695 and recall of 0.795, correctly identifying 210 of 264 Winter test samples. The SVM achieves a Winter F1 of 0.548. Winter's high contrast, cool undertone, and clear coloring produces the most separable representations in both the deep embedding space and the color feature space, making it the most reliably classified season regardless of approach.

The ITA score and a*_mean are the most discriminative features in the SVM pipeline, directly encoding the warm/cool undertone axis. Despite lower overall accuracy, the SVM achieves a per-class recall on Autumn (0.531) that is higher than FaRL's (0.436), demonstrating that explicit undertone features retain a targeted advantage for warm-season detection. Spring is the hardest class for both models — FaRL Spring F1 is 0.464 and SVM Spring F1 is only 0.335 — because Spring's warm-light profile overlaps with both Autumn (warm undertone) and Summer/Winter (lighter tone).

#### Comparison Between FaRL and SVM

FaRL outperforms the SVM on all primary metrics: accuracy (0.564 vs. 0.457), weighted precision (0.567 vs. 0.479), weighted recall (0.564 vs. 0.457), and weighted F1 (0.555 vs. 0.451). The performance gap of 0.107 accuracy points and 0.104 F1 points exceeds the 0.02 significance threshold adopted in this study, confirming that FaRL substantially outperforms the SVM on overall season classification. This indicates that the 7-dimensional handcrafted color feature vector cannot match the representational capacity of a face-specialized Vision Transformer, though the SVM retains competitive per-class recall on Autumn where explicit undertone encoding is most discriminative.

#### Other Findings and Discovered Knowledge

Both models reproduce the Autumn–Winter misclassification pattern. FaRL misclassifies 84 of 259 Autumn samples as Winter. This persistence across different architectures and training configurations indicates that the Autumn–Winter boundary is a structural challenge in the dataset itself — likely arising from overlapping deep, high-contrast features shared by both seasons — rather than a model-specific failure.

Both models also show consistent Spring–Summer confusion, reflecting the shared light-toned characteristics of those two seasons across different undertone temperatures. These recurring inter-season confusion patterns suggest that future work could benefit from a hierarchical classification approach — first classifying undertone temperature (warm vs. cool), then depth (light vs. deep) — to resolve the specific boundaries where current models struggle most.
