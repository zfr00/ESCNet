### Dataset
Due to space limitations of anonymous links, we show part of our training data and validation data in the link https://ufile.io/f/l4j4z, the full and the test data will be open sourced later on.
To make it easier for you to review, we processed the training data into excel format in'.. /data/: 'sample_CCMF.csv' and 'sample_SCMF.csv'. The 'title' column stores the news headlines, and 'text' stores the evidence of the documents crawled, the 'claim_txt_entity' is storing the extracted claim text entity.' doc_txt_entity' is the evidence document text entity, 'claim_img_entity' is the claim image entity, 'doc_img_entity' is the document image entity, 'label' is the label: 0 is Pristine, 1 is Falsified
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

For the full dataset, please contact this email address zfr888@mail.ustc.deu.cn
