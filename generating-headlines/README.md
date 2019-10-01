# Generating News Headlines with RNNs

This project follows a research paper by Konstantin Lopyrev that can be found here: 
* https://arxiv.org/pdf/1512.01712.pdf

Instead of using the English Gigawords dataset, I use Kaggle's All the News dataset.

That dataset is in this repo in 25MB chuncks due to gitHub's file size limit.
You can add them altogether into one tarball with this command:
* cat all-the-news.tar.0* > all-the-news.tar
* tar -xvf all-the-news.tar

