from modules.GlobalVariables import *
from modules.DescriptiveAnalysis import *

from matplotlib import pyplot as plt

import nltk
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# create an instance of SentimentIntensityAnalyzer
sent_analyzer = SentimentIntensityAnalyzer()


class VADER_:
    def __init__(self):
        self = self
    
    def sentiment_analysis(self, run):
        if run:
            data = pd.read_excel("output\\01.Topic_auto_entitled.xlsx", sheet_name="Topics by docs")
            # data = pd.read_excel("output\\00.final_sample.xlsx")
            # data["YEAR"] = data["DATE"].swifter.apply(Descriptive()._bs4_to_year)
            # remove irrelevant or uninterpretable topics
            remove_topics = pd.read_excel("output\\01.Topic_auto_entitled.xlsx", sheet_name="Topics and keywords")
            remove_topics.rename(columns={"Topic":"TOPIC"}, inplace=True)
            merged_data = data.merge(remove_topics, on='TOPIC', how='left')
            filtered_data = merged_data[merged_data['Topic_name'] != 'na']
            # run VADER
            filtered_data["VADER_PREDICT"] = filtered_data["TEXT_PARA_CLEANSED"].swifter.apply(self._predict_sentiment)
            filtered_data.to_excel("output\\03.sentiment_overall.xlsx", index=False)


    def sentiment_trend_analysis(self, run):
        if run:
            data = pd.read_excel("output\\03.sentiment_overall.xlsx")
            subset = self._only_sentence(data)
            dimensions = pd.read_excel("output\\99.summary_semantic&sentiment.xlsx", sheet_name="Sheet1").iloc[:, :2]   
            merged_subset = subset.merge(dimensions, on="Topic_name", how="left")
            merged_subset.to_excel("output\\03.sentiment_overall_filtered.xlsx", index=False)
            # the trend of sentiment ratio
            grouped_data_byYear = merged_subset.groupby(["VADER_PREDICT", "YEAR"]).size().to_frame("NUMBER_PUBLISHED").reset_index()
            data_sentiment_ratio_byYear = self._sentiment_ratio(grouped_data_byYear, "YEAR")
            data_sentiment_ratio_byYear.to_excel("output\\04.sentiment_trend.xlsx", index=False)            
            self._sentiment_plot_by_group(data_sentiment_ratio_byYear, "ratio")   
            # sentiment ratio by topics
            grouped_data_byTopic = merged_subset.groupby(["VADER_PREDICT", "Topic_name"]).size().to_frame("NUMBER_PUBLISHED").reset_index()
            data_sentiment_ratio_byTopic = self._sentiment_ratio(grouped_data_byTopic, "Topic_name")
            merged_ratio = data_sentiment_ratio_byTopic.merge(dimensions, on="Topic_name", how="left")
            merged_ratio = merged_ratio[['Dimensions', 'Topic_name', 'No_positive', 'SENT_RATIO']]
            merged_ratio.to_excel("output\\05.sentiment_byTopic.xlsx", index=False)
            # the trend of sentiment ratio by dimensions
            # Assuming "positive" corresponds to 1, "neutral" to 0, and "negative" to -1
            merged_subset['VADER_PREDICT'] = merged_subset['VADER_PREDICT'].map({'positive': 1, 'neutral': 0, 'negative': -1})
            # Group by 'YEAR' and 'TOPIC' and calculate the positive ratio
            sentiment_ratio_byY_T = merged_subset[merged_subset['VADER_PREDICT'] == 1].groupby(['YEAR', 'Dimensions']).size() / merged_subset.groupby(['YEAR', 'Dimensions']).size()
            toDF = sentiment_ratio_byY_T.to_frame().reset_index()
            toDF.rename(columns={0:"Positive_Ratio"}, inplace=True)
            toDF.to_excel("output\\06.sentiment_byYear&Dimension.xlsx", index=False)
            self._plot_ratio(toDF)

            return merged_subset

    def _only_sentence(self, data): # to have rows with only sentences ... assuming that a sentence consists of at least ten words
        # Calculate the word count for each row and create a new column 'Word_Count'
        data["word_count"] = data['TEXT_PARA_CLEANSED'].apply(lambda x: len(str(x).split()))
        # Filter out rows with less than 10 words
        df_filtered = data[data['word_count'] >= 10]
        # Drop the 'Word_Count' column if you no longer need it
        df_filtered = df_filtered.drop(columns=['word_count'])        
        return df_filtered

    def _predict_sentiment(self, data):    
        output = sent_analyzer.polarity_scores(data)
        return self._sentiment_output(output)
    
    def _sentiment_output(self, value):
        polarity = "neutral"
        if value["compound"] >= 0.05:
            polarity = "positive"        
        elif value["compound"] <= -0.05:
            polarity = "negative"
        return polarity

    def _sentiment_ratio(self, data, column_): # column_ = YEAR or Topic_name
        if column_ == "YEAR":
            start_year = data[column_].min()
            end_year = data[column_].max()
            column_list = list(np.arange(start_year, end_year+1))
        else: # column_ == "Topic_name"
            column_list = data['Topic_name'].unique()
        sent_ratio = []
        positive_no = []
        # sentiment ratio in a certain year
        for i in range(len(column_list)):
            # number of positive articles in a certain year
            try:
                data_pos = data.loc[(data["VADER_PREDICT"] == "positive") & (data[column_] == column_list[i])]["NUMBER_PUBLISHED"].item()
            except: # ValueError: can only convert an array of size 1 to a Python scalar
                data_pos = 0
            # # number of negative articles in a certain year
            # try:
            #     data_neg = data.loc[(data["VADER_PREDICT"] == "negative") & (data["YEAR"] == column_list[i])]["NUMBER_PUBLISHED"].item()
            # except: # ValueError: can only convert an array of size 1 to a Python scalar
            #     data_neg = 0
            # if (data_neg == 0):
            #     sent_ratio.append(np.NAN)
            # else:
            #      sent_ratio.append(data_pos / data_neg)
            # number of articles in a certain year
            count = data.loc[data[column_] == column_list[i], "NUMBER_PUBLISHED"].sum()
            positive_no.append(data_pos)
            sent_ratio.append(round(data_pos / count, 3))
        # make a dataframe to match sentiment ratio and year
        df_output = pd.DataFrame((zip(column_list, positive_no, sent_ratio)), columns = [column_, "No_positive", "SENT_RATIO"])
        return df_output        

    def _sentiment_plot_by_group(self, data, type_): #type_ = ratio or number
        if type_ == "ratio":
            data.set_index("YEAR", inplace = True)
            data["SENT_RATIO"].plot(label = "AVs", color = "red")
            plt.title("Positive Ratio per Year")
            plt.ylabel("Positive/Total")
            plt.xticks(np.arange(min(data.index), max(data.index)+1, 3.0))
            plt.legend()
            plt.savefig("output\\11.trend_sentiment_ratio.jpg")
            plt.clf()
        # else:
        #     fig, axs = plt.subplots(ncols = 2)
        #     sns.lineplot(x="YEAR", y="NUMBER_PUBLISHED", data=data, hue="VADER_PREDICT", palette= "Accent", ax = axs[1])
        #     axs[1].set_title("Autonomous Vehicle")
        #     plt.savefig(join(os.getcwd(), "output", filename))
        #     plt.clf()

    def _plot_ratio(self, data):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(data)), data['Positive_Ratio'])
        plt.xlabel('Year (Topic)')
        plt.ylabel('Positive Ratio')
        plt.title('Annual Trend of Positive Ratio by Dimensions')

        # Convert 'YEAR' to a string and concatenate with 'TOPIC'
        xtick_labels = data.apply(lambda row: f"{row['YEAR']} ({row['Dimensions']})", axis=1)
        plt.xticks(range(len(data)), xtick_labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.tight_layout()
        plt.legend()
        plt.savefig("output\\12.trend_sentiment_byTopic&Year.jpg")
        plt.clf()        