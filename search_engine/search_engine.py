#!/usr/bin/env python

import time
from collections import defaultdict
from array import array
import math
import numpy as np
import pandas as pd
import collections
from numpy import linalg as la

import json
import datetime
from urllib3.exceptions import ProtocolError
import time

from util import create_index_tfidf_tp
from util import search_tf_idf
from util import search_itp

with open("../data/processed_data_2.json", "rb") as f:
    data = f.readlines()
    data = [json.loads(str_) for str_ in data]

df_tweets = pd.DataFrame.from_records(data)


def main():
    print("Welcome to the Twitter search engine. Please wait so we can load all data")
    print("Creating index tf idf....(30s aprox)")
    start_time = time.time()
    index, tf, df, idf, tp, itp = create_index_tfidf_tp(df_tweets)
    print(
        "Total time to create the index: {} seconds".format(
            np.round(time.time() - start_time, 2)
        )
    )
    ranked_docs = []
    while True:

        print("Welcome to the search engine:\n")
        print("Press 1 to search with tf-idf:\n")
        print("Press 2 to search with tp (term popularity):\n")
        print("Press 0 to exit (term popularity):\n")
        option = input()

        print("\n")
        invalid = False
        if option == "1":
            print("Insert query tf-idf:\n")
            query = input()
            ranked_docs = search_tf_idf(query, index, tf, idf)
        elif option == "2":
            print("Insert query tp:\n")
            query = input()
            ranked_docs = search_itp(query, index, tf, itp)
        elif option == "0":
            print("See you again")
            exit(0)
        else:
            print("Insert a valid query")
            invalid = True
        if len(ranked_docs) == 0:
            invalid = True
            print(
                "Sorry, we did not found tweets for this query, please select a score and insert a query"
            )
        if invalid == False:
            print("Insert your query:\n")

            top = 10
            print(query)
            print(
                "\n======================\nTop {} results out of {} for the seached query:\n".format(
                    top, len(ranked_docs)
                )
            )
            for d_id in ranked_docs[:top]:
                text = df_tweets["text"][d_id]
                screen_name = df_tweets["user"][d_id]["screen_name"]
                tweetid = df_tweets["id"][d_id]
                date = df_tweets["created_at"][d_id]
                hastags = df_tweets["hashtags"][d_id]
                likes = df_tweets["likes"][d_id]
                retweets = df_tweets["retweets"][d_id]

                url = "https://twitter.com/{}/status/{}".format(screen_name, tweetid)
                print(
                    "TWEET: {} | USERNAME: {} | DATE: {} | HASHTAGS: {} | LIKES: {} | RETWEETS: {} | URL: {}\n".format(
                        text,
                        screen_name,
                        date,
                        hastags,
                        likes,
                        retweets,
                        url,
                    )
                )


if __name__ == "__main__":
    main()