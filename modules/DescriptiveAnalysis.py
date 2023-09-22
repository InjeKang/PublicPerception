from modules.GlobalVariables import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Descriptive:
    def __init__(self):
        self = self
    
    def annual_trend(self, run):
        if run:
            data = pd.read_excel("output\\01.Topic_auto_entitled.xlsx", sheet_name="Topics by docs")
            # data["YEAR"] = data["DATE"].swifter.apply(self._bs4_to_year)
            # remove duplicate ID_DOC to count the number of articles
            subset_remove_duplicate = data.drop_duplicates(subset=["ID_DOC"])
            # trend analysis
            trend_result = subset_remove_duplicate.groupby(["KEYWORD", "YEAR"]).size().to_frame("NUMBER_PUBLISHED").reset_index()
            # plot
            self._plot_by_group(trend_result, "NUMBER_PUBLISHED")
            # save the result
            trend_result.to_excel("output\\02.annual_trend.xlsx", index=False)
    
    def _plot_by_group(self, data, column_):
            # set the column YEAR as Index
            trend_ = data.copy()
            trend_.set_index("YEAR", inplace = True)
            # plot time series
            trend_.loc[trend_["KEYWORD"] == "AD"][column_].plot(label = "AVs", color = "red")
            plt.title("News Articles per Year")
            plt.ylabel("Number of published news articles")
            plt.xticks(np.arange(min(trend_.index), max(trend_.index)+1, 3.0))
            plt.legend()
            plt.savefig("output\\10.annual_trend.jpg")
            plt.clf()

    def _bs4_to_year(self, x):
        date_ = str(x.encode("utf-8"))
        year_ = int(re.findall(r"\d{4}", date_)[0])
        return year_



