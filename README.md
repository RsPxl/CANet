# CANet
Intelligent Remote Sensing of Green Tide

# Usage Guide
This project developed an intelligent model for MODIS green tide detection, which takes Rrc data as input and outputs classifications such as green tides, clouds, and seawater.
Readers can refer to the paper and run the code according to the order indicated by the code prefixes to perform green tide detection and generate the dataset. 
If you have any questions, please feel free to contact the authors.

# Environment
- Python                     3.6.2
- tensorflow-gpu             2.4.0
- Keras                      2.4.3
- GDAL                       2.4.1
- numpy                      1.19.5


# Pipeline & Scripts

01-Preprocess.py: Image band combination.

    Input: Data/DataSet (Original satellite images).
    Output: Data/DataSet_BandCombine (Satellite images with FAI).

02-preprocess_train.py: Crop training patches.

    Input: Data/DataSet_BandCombine, Data/DataSet_BandCombine_Label (Label).
    Output: Data/DataSet_BandCombine_Train/Img_SampleSet (Image patches), Data/DataSet_BandCombine_Train/Lab_SampleSet (Label patches).

03-Del_background.py: Filter green tide samples.

    Input:  Data/DataSet_BandCombine_Train/Img_SampleSet, Data/DataSet_BandCombine_Train/Lab_SampleSet.
    Output: Data/DataSet_BandCombine_Train/Img_SampleSet_DelBackground (Green-tide-containing image patches), Data/DataSet_BandCombine_Train/Lab_SampleSet_DelBackground (Green-tide-containing label  patches).

04-main.py: Train model.

    Input: Data/DataSet_BandCombine_Train/Img_SampleSet_DelBackground, Data/DataSet_BandCombine_Train/Lab_SampleSet_DelBackground.
    Output: logs/CANet.h5 (Final model).

05-main-test-overlap.py: Test model.

    Input: Data/DataSet_BandCombine_Test (Satellite images with FAI for testing).
    Output: Data/DataSet_BandCombine_Test_Result (Test results).
    Using: logs/CANet.h5

06-Image_Filter.py: Remove strip noise.

    Input: Data/DataSet_BandCombine_Test_Result.
    Output: Data/DataSet_BandCombine_Test_Result_Filter (Denoised results).

07-Reprocess_Unmixing.py: Pixel unmixing.

    Input: Data/DataSet_BandCombine_Test, Data/DataSet_BandCombine_Test_Result_Filter.
    Output: Data/DataSet_BandCombine_Test_Result_Filter_Unmixing (Sub-pixel abundance maps).

08-SpatialSample.py: Resampled to a consistent spatial resolution.

    Input: Data/DataSet_BandCombine_Test_Result_Filter_Unmixing.
    Output: Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample (Resampling results).

09-Image_Alignment.py: Consistent Image Dimensions.

    Input: Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample, Data/Tar_zeros.tif (Standardized Structured Data).
    Output: Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment (Consistent image dimensions).

10-Img_Com_Week_4km.py, 10-Img_Com_Month_4km.py, 10-Img_Com_Year_4km.py: Multi-scale Temporal Image Compositing.
    There is no specific order; the code can be executed according to the requirements of the Image composition.

    Input: Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment.
    Output: Data/DataSet_BandCombine_Test_Result_Filter_Unmixing_Sample_Alignment_Composition (Image composition results).

model.py: Model architecture.

read_gdal.py: GDAL image I/O.

logs/CANet.h5: Trained Model.

Data/Tar_zeros.tif: Standardized Data Format for Dataset Generation.

# License

This dataset is released under the Creative Commons CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

https://creativecommons.org/publicdomain/zero/1.0/

The data can be freely used, modified, and distributed without restriction.
