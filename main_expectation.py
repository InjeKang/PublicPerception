import datetime
import pytz

from modules.GlobalVariables import *
from modules.SemanticAnalysis import *
from modules.DescriptiveAnalysis import *
from modules.SentimentAnalysis import *

def main():
    # Print the current time in South Korea
    current_time_korea = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    print("Started:", current_time_korea.strftime("%Y-%m-%d %H:%M"))

    run_ = {"bert_coherence_report": False, "bert_select_model":False,
            "annual_trend":False, "sentiment_analysis":False, "sentiment_trend":True}

    # coherence report
    result = BERTprocess().create_coherence_report(start=5, end=50, run=run_["bert_coherence_report"])

    # run BERTopic for semantic analysis
    result = BERTprocess().select_model(run=run_["bert_select_model"])

    # descriptive
    result = Descriptive().annual_trend(run=run_["annual_trend"])

    # run VADER for sentiment analysis...run after entitling the topics
    result = VADER_().sentiment_analysis(run=run_["sentiment_analysis"])
    result = VADER_().sentiment_trend_analysis(run=run_["sentiment_trend"])

    

    current_time_korea_finished = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    print("Ended:", current_time_korea_finished.strftime("%Y-%m-%d %H:%M"))
    return result




if __name__ == "__main__":
    main()
