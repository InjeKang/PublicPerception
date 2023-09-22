import pandas as pd
import numpy as np
import re
import os
import swifter
from modules.variables import *
from tqdm import trange
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup
from os.path import join



def read_data(filename, sheet_="Sheet1"):
    default_path = os.getcwd()
    input_path = join(default_path, "data")        
    # change default directory to read data
    os.chdir(input_path)
    # read excel file
    if filename.endswith("xlsx"):
        data = pd.read_excel(filename, engine="openpyxl", sheet_name = sheet_)
    # read pickle file
    else:
        data = pd.read_pickle(filename)        
    # reset default directory
    os.chdir(default_path)
    return data


def multiprocess(datalist, target_func):
    pool = Pool()
    results = []
    ITERATION_COUNT = cpu_count() - 1
    count_per_iteration = len(datalist) / float(ITERATION_COUNT)
    for i in trange(ITERATION_COUNT):
        list_start = int(count_per_iteration * i)
        list_end = int(count_per_iteration * (i+1))
        results.append(pool.apply_async(target_func, (datalist[list_start:list_end],)))
    pool.close()
    pool.join()
    results_val = [results[i].get() for i in range(len(results)) if results[i].successful()]
    return results_val


def xml_to_dataframe(datalist):
    df_data = {key: val for key, val in zip(COLS, ROWS)}
    for type, doc in datalist:
        soup = BeautifulSoup(doc, "xml")
        for key, css in zip(COLS, SELS):
            if key == "KEYWORD":
                new_val = type.upper()
            else:
                try:
                    new_val = soup.select_one(css).string
                except:
                    new_val = ""
            df_data[key].append(new_val)
    return pd.DataFrame(df_data)


def extract_body(df):
    # df["TEXT_XML"] = df.TEXT_XML.swifter.apply(lambda x: RE_DUPTAG.sub("</p>", x))
    df["TEXT_PARA"] = df.TEXT_XML.swifter.apply(lambda x: list(BeautifulSoup(x, "lxml").select("body > p")))
    df["TEXT_FULL"] = df.TEXT_PARA.swifter.apply(lambda x: RE_WS.sub("", RE_TAG.sub("", " ".join(map(str, x)).replace("\\", "/"))).strip())
    return df


def parse(df, parse_run):
    input_list = [(kw, xml) for kw, xml in zip(list(df.KEYWORD), list(df.XML))]
    if parse_run:
        # results_val = multiprocess(input_list, xml_to_dataframe)
        df = xml_to_dataframe(input_list)
        result = extract_body(df)
        save_data(result, "1.data_parsed")
    else:
        result = pd.read_pickle(join(BASE_PATH, "input", "1.data_parsed", "1.data_parsed"))
    return result


def split(df, split_run):
    if split_run:
        result = df.explode("TEXT_PARA")
        ind_freq = result.index.value_counts().sort_index()
        result["ID_DOC"] = result.index
        result["ID_SUB"] = np.concatenate([list(range(ind_freq[i], 0, -1)) for i in range(len(ind_freq))])
        result["ID_PARA"] = [f"{result.index[i]}-{result.ID_SUB.values[i]}" for i in range(result.shape[0])]
        result.index = range(result.shape[0])
        save_data(result, "2.data_exploded")
    else:
        result = pd.read_pickle(join(BASE_PATH, "input", "2.data_exploded", "2.data_exploded"))
    return result            


def cleanse(df, cleanse_run):
    if cleanse_run:
        df["TEXT_PARA_CLEANSED"] = df.TEXT_PARA.swifter.apply(lambda x: RE_WS.sub("", RE_TAG.sub("", str(x))))
        result = df
        save_data(result, "3.data_cleansed")
    else:
        result = pd.read_pickle(join(BASE_PATH, "input", "3.data_cleansed", "3.data_cleansed"))
    return result


def filter(df, filter_run):
    if filter_run:
        result = df[["ID_DOC", "ID_SUB", "ID_PARA", "KEYWORD", "DATE", "TITLE", "TEXT_FULL", "TEXT_PARA_CLEANSED"]]
        result = result.drop(df[df.TEXT_PARA_CLEANSED.map(len) <= 2].index)
        result.reset_index(drop=True, inplace=True)
        save_data(result, "4.data_filtered")
    else:
        result = pd.read_pickle(join(BASE_PATH, "input", "4.data_filtered", "4.data_filtered"))
    return result


def save_data(df, file_name, type="input"):
    directory = join(BASE_PATH, type, file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(join(directory, f"{file_name}.csv"), encoding="utf-8", index=False)
    df.to_excel(join(directory, f"{file_name}.xlsx"), encoding="utf-8")
    df.to_pickle(join(directory, f"{file_name}"))
    return


