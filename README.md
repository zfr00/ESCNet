## Docker
Our code runtime environment is created by docker and we provide you with two dockerfiles located at . /dockertmp, where you can build the image from scratch. 

## Dataset
Due to space limitations of anonymous links, we show part of our training data and validation data, the full and the test data will be open sourced later on.
To make it easier for you to navigate, we processed the training data into excel format in'.. /data/: 'sample_CCMF.csv' and 'sample_SCMF.csv'. The 'title' column stores the news headlines, and 'text' stores the evidence of the documents crawled, the 'claim_txt_entity' is storing the extracted claim text entity.' doc_txt_entity' is the evidence text entity, 'claim_img_entity' is the claim image entity, 'doc_img_entity' is the evidence image entity, 'label' is the label: 0 is Pristine, 1 is Falsified

The training image data format is as follows, corresponding to 'image' in the csv.
```
data/
├── collected_img/
│   ├── img_fake/
│   │   ├── claim_fake/
│   │   └── document_fake/
│   ├── img_real/
│   │   ├── claim_real/
│   │   └── document_real/
└── synthetic_img/
    ├── img_fake/
    │   ├── claim_fake/
    │   └── document_fake/
    └── img_real/
        ├── claim_real/
        └── document_real/
```

For convenience, the full dataset is linked at https://drive.google.com/drive/folders/1zCd3rgUvQ3i1Gt6_d8yvefUNwKR7sdAE?usp=drive_link

## Crawler Code
It would be an honor if our scripts for building datasets could inspire readers to repurpose them for making datasets in languages other than Chinese. Due to space and time constraints, we have shared part of the crawler script for you in ./crawler/, where 'fake_excel' contains the intermediate data for our crawled dataset.'url_list' are some of the website links used in crawling the data. Different scripts are labeled with the corresponding websites, such as 'QQ', 'mingcha'. 
The script that contains the 'search_image' information contains the process of crawling images.

!!!You can visit https://huggingface.co/ckiplab/bert-base-chinese-ner to extract various types of Chinese entities and perform an image search to build your own SCMF dataset!

It may contain some Chinese characters, so sorry for the inconvenience of reading it!
To avoid copyright infringement, the complete end-to-end crawler script will be released in an open source version.

## Quick Start (if you are interested)
In order to avoid anonymizing the download speed of the disk to your mood, we stored the training data in the form of a pickle and updated the training code for you
After getting the pre-training bert-chinese ready and your_path modified correctly
- have a try
    ```
    python train.py
    ```

## Generate Your Own Traing Data
If you want to understand the full training process, read '.. /entity/README.md', after making the relevant preparations
- pleace run
    ```
    python generate_pickle_for_review.py
    ```
