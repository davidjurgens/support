# GenderPerformr

## Intro

Supportr is the code release for labeling replies on the relative
supportiveness, based on the paper `It’s going to be okay: Measuring Access to
Support in Online Communities` by Zijian Wang and David Jurgens (in proceedings
of EMNLP 2018).

See the project website for full details, including contact information.

## Install 

The code is not yet setup as a python package but the code can be run using the `example.sh` script in the main directory.  

### Dependencies

The `setup.py` file lists the model dependencies.  Aside from these python
packages, the code requires the Google news word2vec vectors in the `resources/`
directory.  The model described in the paper uses LIWC categories as features.
LIWC is a commercial product and cannot be redistributed.  However, the Empath
library is known to closely approximate the categories and the code is setup to
Just Work™ if you don't have LIWC purchase.  If you _do_ have LIWC, it should be
put in `resources/lexicons/` and named `en_liwc.txt`.  Additional tests are
needed to report performance when LIWC is not used.

## Data

Included in this release is the aggregrated and labeled training data for the
paper, which can be used to reproduce the results of the paper or improve the
classifier.


## Release History

* 0.1 Initial code release

The next step is to package the code up as a module to allow easier
classification.  Pull requests welcome.

