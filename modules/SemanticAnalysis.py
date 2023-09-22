from modules.GlobalVariables import *
from modules.DescriptiveAnalysis import *

import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from bertopic import BERTopic


class BERTprocess:
    def __init__(self):
        self = self

    # create a coherence report
    def create_coherence_report(self, start, end, run):
        if run:
            data = read_data("5.data_cleansed")
            # select only AV related articles
            data_onlyAD = data[data["KEYWORD"].str.contains("AD")]
            # to remove nearly-empty documents from https://github.com/MaartenGr/BERTopic/issues/90
            mask = data_onlyAD["TEXT_PARA_CLEANSED"].swifter.apply(lambda x: len(x)>=10)
            subset = data_onlyAD[mask]
            subset.reset_index(drop=True, inplace=True)
            subset.to_excel("output\\6.final_sample.xlsx", index=False)
            coherence_df = pd.DataFrame([self._get_coherence(subset, i) for i in range(start, end+1)], columns=["N_TOPICS", "COHERENCE"])
            coherence_df.to_excel("output\\coherence\\01.coherence_by_topics.xlsx")
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(coherence_df['N_TOPICS'], coherence_df['COHERENCE'], marker='o', linestyle='-', color='b')
            plt.title('Relationship between n_topics and Coherence')
            plt.xlabel('n_topics')
            plt.ylabel('Coherence')
            plt.grid(True)
            plt.savefig("output\\coherence\\01.coherence_by_topics.png")
            plt.clf() 


    def _get_coherence(self, df, num_topics):
        docs = list(df["TEXT_PARA_CLEANSED"])
        vectorizer_model = CountVectorizer(stop_words = "english")
        topic_model = BERTopic(vectorizer_model=vectorizer_model, 
                                n_gram_range = (1, 2),
                                top_n_words = 20, 
                                nr_topics=num_topics, 
                                # diversity=0.5,
                                low_memory=True)
        topics, probs = topic_model.fit_transform(docs)
        result = pd.concat([df, pd.DataFrame({"TOPIC": topics, "PROB": probs})], axis=1)
        cleaned_docs = topic_model._preprocess_text(docs)

        # Extract vectorizer and tokenizer from BERTopic
        vectorizer = topic_model.vectorizer_model
        tokenizer = vectorizer.build_tokenizer()

        # Extract features for Topic Coherence evaluation
        tokens = [tokenizer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                    for topic in range(len(set(result.TOPIC))-1)]
        # Remove lists with empty elements, ["","","",...] or ["editor", "", "",...] >> not suitable to find the optimal topic number
        filtered_topic_words = [sublist for sublist in topic_words if not all(item == "" for item in sublist)]
        filtered_topic_words = [[item for item in sublist if item != ""] for sublist in filtered_topic_words]
        # Remove lists with only one element
        filtered_topic_words = [sublist for sublist in filtered_topic_words if len(sublist) >= 2]
        # Evaluate
        # try:
        coherence_model = CoherenceModel(topics=filtered_topic_words, 
                                        texts=tokens, 
                                        corpus=corpus,
                                        dictionary=dictionary, 
                                        coherence='c_v')
        coherence = coherence_model.get_coherence()
        topic_model.save(f"output\\coherence\\00.BERTopic_model_{num_topics}")
        return num_topics, coherence
        # except:
        #     return num_topics, np.nan
    
    def select_model(self, run):
        if run:
            data = read_data("5.data_cleansed")
            data["YEAR"] = data["DATE"].swifter.apply(Descriptive()._bs4_to_year)
            data_onlyAD = data[data["KEYWORD"].str.contains("AD")]
            # to remove nearly-empty documents from https://github.com/MaartenGr/BERTopic/issues/90
            mask = data_onlyAD["TEXT_PARA_CLEANSED"].swifter.apply(lambda x: len(x)>=10)
            subset = data_onlyAD[mask]
            subset.reset_index(drop=True, inplace=True)
            subset.to_excel("data\\6.final_sample.xlsx", index=False)            
            # BERTopic with number of topics extracted automatically
            vectorizer_model = CountVectorizer(stop_words = "english")            
            topic_model = BERTopic(vectorizer_model=vectorizer_model, 
                                    n_gram_range = (1, 2),
                                    top_n_words = 20, 
                                    nr_topics="auto", min_topic_size = 50,
                                    # diversity=0.5,
                                    low_memory=True)
            docs = list(subset["TEXT_PARA_CLEANSED"])
            topics, probs = topics, probs = topic_model.fit_transform(docs)
            topics_doc = pd.concat([subset, pd.DataFrame({"TOPIC": topics, "PROB": probs})], axis=1)
            topics_kw = topic_model.get_topic_info()
            topics_kw["Keywords"] = [" | ".join([kw for kw, prob in topic_model.get_topic(i)]) for i in list(topics_kw.Topic)]
            with pd.ExcelWriter("output\\01.Topic_auto.xlsx") as writer:
                topics_doc.to_excel(writer, sheet_name="Topics by docs", index=False)
                topics_kw.to_excel(writer, sheet_name="Topics and keywords", index=False)                
                    
            # result_coherence = pd.read_excel("output\\coherence\\01.coherence_by_topics.xlsx")
            # sorted_df = result_coherence.sort_values(by='COHERENCE', ascending=False)  # Sort DataFrame by COHERENCE in descending order
            # options = sorted_df['N_TOPICS'].head(5).tolist()  # Select the top five N_TOPICS values
            # for num_topic in options:
            #     topic_model = BERTopic.load(f"output\\coherence\\00.BERTopic_model_{num_topic}")
            #     topics, probs = topic_model._map_predictions(topic_model.hdbscan_model.labels_), topic_model.hdbscan_model.probabilities_
            #     topics_doc = pd.concat([subset, pd.DataFrame({"TOPIC": topics, "PROB": probs})], axis=1)
            #     topics_kw = topic_model.get_topic_info()
            #     topics_kw["Keywords"] = [" | ".join([kw for kw, prob in topic_model.get_topic(i)]) for i in list(topics_kw.Topic)]
                
            #     # create a excel writer object
            #     with pd.ExcelWriter(f"output\\01.Topic_{num_topic}.xlsx") as writer:
            #         topics_doc.to_excel(writer, sheet_name="Topics by docs", index=False)
            #         topics_kw.to_excel(writer, sheet_name="Topics and keywords", index=False)                
