import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("volume2/testDB.db")

print('reading in df')
df = pd.read_sql("""
                 SELECT doi,editors,ab_toks from PLOS_ALL
                 JOIN PLOS_ALL_tok USING(doi)
                 """, conn)

print('grouping abstracts')
df_ed = df.groupby('editors')['ab_toks'].apply(' '.join)

print(df_ed.head())

print('saving to db')
df_ed.to_sql('ed_concat_abtoks', conn, if_exists="replace")

print('complete')