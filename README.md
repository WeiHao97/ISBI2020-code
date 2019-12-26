# ISBI 2020: Learning Amyloid Pathology Progression from Longitudinal PiB-PET Images in Preclinical Alzheimer’s Disease
Author: Wei Hao, Nicholas M. Vogt, Zihang Meng, Seong Jae Hwang, Rebecca L. Koscik, Sterling C. Johnson, Barbara B. Bendlin and Vikas Singh
_____________________________
## Abstarct:
Amyloid accumulation is acknowledged to be a primary pathological event in Alzheimer’s disease (AD). The literature suggests that the transmission of amyloid happens along neural pathways as a function of the disease process (prionlike transmission), but the pattern of spread in the preclinical stages of AD is still poorly understood. Previous studies have used diffusion processes to capture amyloid pathology propagation using various strategies and shown how future time-points can be predicted at the group level using a population-level structural connectivity template. But the structural pathways are different between distinct subjects, and the current literature is unable to provide individual-level pathology propagation. We develop a trainable network diffusion model that infers the propagation dynamics of amyloid pathology, conditioned on individual-level structural connectivity network. Our model on longitudinal amyloid pathology estimates in 16 gray matter (GM) regions known to be affected by AD (N = 112) individuals, measured using Pittsburgh Compound B (PiB) positron emission tomography at 3 different time points for each subject. Experiments show that our model outperforms inference based on group-level trends for predicting future time points data (using individual-level structural connectivity network). For group-level analysis, we find significant group parameter differences (via permutation testing) between APOE positive and APOE negative subjects.
_____________________________
## File Structure:
1. assets	
2. build/lib/torchdiffeq
3. dist	
4. pib
    - put CSVs and TXTs here
5. torchdiffeq.egg-info
6. torchdiffeq	
7. README.md
8. v1
9. v2
10. visualization
    - use script_fiber_bundles_v2_040219 for visualization
_____________________________
## Utilization and dependency:
1. use Data_Preparation.ipynb to generate clean data for general use
2. use Model_train_and_evaluation.ipynb to generate prediction model in different time points and evaluation
3. use pre_train.py to generate pretrain model for APOE_Analysis_Data_Generator.ipynb
4. use APOE_Analysis_Data_Generator.ipynb to generate random testing data for apoe analysis
5. use APOE_Analysis.ipynb to perform apoe analysis
