The implementation is made of three folders:

    1. Character Compositionality(Chise): The txt file in this folder is the original file of Chise dataset. 
    The csv file is the file after processing. The ipynb file is the code on processing Chise dataset.

    2. MSRA: Experiments on MSRA dataset. A folder named Scheme_1_100_50, which means this model is based on scheme 1 in our paper, 
    and uses 100 dimensions of character embedding and 50 dimensions of radical embedding. Every folder for experiments has a csv file, which contains
    the performance of the models. 

    tokenrzse.ipynb is the process for tokenrizing the tags, Chinese characters and Chinese radicals.

    result_analysis.ipynb shows the best performance of every model.

    3. People'sDAily: All same with MSRA but the dataset is People'sDAily.


A little amount of code is reused from a NER tutorial. Here is the reference:

    https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/


All the trained models are removed from this file, while the csv files can show the performance of models.

If you have any confusion, please e-mail me: ruiyan.gong2@durham.ac.uk
If any questions after 30th June (my DU mailbox will be expired), please email me: 290511458@qq.com