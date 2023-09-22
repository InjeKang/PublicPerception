import pandas as pd
import re
from os.path import join

# Input paths
BASE_PATH = r"./data/"
INPUT_PATH = join(BASE_PATH, "input", "0.data_raw", f"nyt")
OUTPUT_PATH = join(BASE_PATH, "output")
# XMLS_DOCS = [list(pd.read_csv(input_path).XLM) for input_path in INPUT_PATHS]

# Dataframe elements
CSS_GOID = "RECORD > GOID"
CSS_LANG = "RECORD > Obj > Language > RawLang"
CSS_TITLE = "RECORD > Obj Title"
CSS_DATE = "RECORD > Obj NumericDate"
CSS_PUBLISHER = "RECORD > DFS > PubFrosting > Title"
CSS_EDITION = "RECORD > DFS > PubFrosting > Edition"
CSS_TEXT_XML = "RECORD > TextInfo > Text"
COLS = ["KEYWORD", "GOID", "LANG", "TITLE", "DATE", "PUBLISHER", "EDITION", "TEXT_XML"]
ROWS = [[] for i in range(len(COLS))]
SELS = ["", CSS_GOID, CSS_LANG, CSS_TITLE, CSS_DATE, CSS_PUBLISHER, CSS_EDITION, CSS_TEXT_XML]

# Regex patterns
RE_DUPTAG = re.compile(u"</p>[\s\t\n]{0,}</p>")
RE_TAG = re.compile(u"<.*?>")
RE_WS = re.compile(u"\s{2,}")