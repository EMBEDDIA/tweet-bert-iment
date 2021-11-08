# tweet-bert-iment
Sentiment analysis of tweets based on BERT models

The repository contains training code used to train models for sentiment analysis of short texts, as well as the code for a simple dockerized service and an ELG-compatible service (using ELG SDK).

The sentiment analysis models use transformer models and were finetuned using labelled tweets (Mozetič, Igor; Grčar, Miha and Smailović, Jasmina, 2016, Twitter sentiment for 15 European languages, Slovenian language resource repository CLARIN.SI, http://hdl.handle.net/11356/1054.)

The trained models for English, Croatian, Russian, and Slovene were finetuned on bert-base-cased, Bertić, RuBERT, and SloBERTa models, respectively, and are available on HuggingFace (https://huggingface.co/EMBEDDIA)

The services for these four models are publicly available on ELG, and as docker images on dockerhub (https://hub.docker.com/u/matejulcar).
