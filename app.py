import os
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from collections import Counter
from dash import Dash, dcc, html, Input, Output, dash_table
import nltk

nltk.data.path.append("./nltk_data")
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def split_title(title_info, keyword_stat, init_stat, conf_idx):
    interpunctuations = [
        ',',
        '.',
        ':',
        ';',
        '?',
        '(',
        ')',
        '[',
        ']',
        '&',
        '!',
        '*',
        '@',
        '#',
        '$',
        '%',
    ]
    stopwords_deep_learning = [
        'learning',
        'network',
        'neural',
        'networks',
        'deep',
        'via',
        'using',
        'convolutional',
        'single',
    ]

    keyword_list = []
    wnl = WordNetLemmatizer()

    for i in tqdm(range(len(title_info))):
        word_list = word_tokenize(title_info[i].lower())
        tagged_word_list = pos_tag(word_list)

        lemmas_sent = []
        for tag in tagged_word_list:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        word_list = list(set(lemmas_sent))

        word_list_cleaned = []
        for word in word_list:
            # remove stopwords
            if (
                word not in interpunctuations
                and word not in stopwords.words('english')
                and word not in stopwords_deep_learning
            ):
                # save the word and the sentence id
                word_list_cleaned.append((word, i))
        keyword_list += word_list_cleaned

    keyword_counter = Counter([item[0] for item in keyword_list])
    print(f"{len(keyword_counter)} keywords are extracted from the paper titles")

    for k in keyword_counter:
        if k not in keyword_stat:
            keyword_stat[k] = copy.deepcopy(init_stat)
        keyword_stat[k][0][conf_idx] = keyword_counter[k]

    for item in keyword_list:
        if item[0] in keyword_counter:
            keyword_stat[item[0]][1][conf_idx].add(item[1])


def generate_keywd_list(conference_info, keyword_idx):
    conf_list = [*conference_info.keys()]
    tmp_df_list = list()
    for conf_idx, set_idx in enumerate(keyword_idx):
        if len(set_idx) == 0:
            continue
        df = conference_info[conf_list[conf_idx]]
        tmp_df = df.loc[sorted([*set_idx])]
        tmp_df_list.extend(tmp_df.to_dict('records'))
    return [tmp_df_list]


def load_paper_info(args, file_path="./stat/papers.xlsx"):
    info = pd.read_excel(file_path, sheet_name=None)
    for k, v in info.items():
        info[k].insert(0, "Conference", k)
        for i in range(len(info[k])):
            info[k].iloc[i, 3] = (
                f"[[InfoLink]({info[k].iloc[i, 3].replace('/papers/','/html/').replace('.pdf','.html')})] "
                + f" [[PDFLink]({info[k].iloc[i, 3]})]"
            )
    sheet_names = [*info.keys()]

    if args.update_cache or not os.path.exists("./stat/cache.npy"):
        init_stat = [
            [0 for _ in range(len(sheet_names))],
            [set() for _ in range(len(sheet_names))],
        ]
        keyword_dict = dict()

        for idx, (k, v) in enumerate(info.items()):
            print(k)
            title = v["Title"].tolist()
            split_title(title, keyword_dict, init_stat, idx)

        # sort keywords by frequency
        keyword_dict = {
            k: v
            for k, v in sorted(keyword_dict.items(), key=lambda item: -sum(item[1][0]))
        }
        np.save("./stat/cache.npy", keyword_dict)
    else:
        keyword_dict = np.load("./stat/cache.npy", allow_pickle=True).item()

    tendency_of_keywd = dict()
    for k, v in keyword_dict.items():
        tendency_of_keywd[k] = v[0]
    tendency_of_keywd['Conference'] = sheet_names
    df = pd.DataFrame.from_dict(
        tendency_of_keywd, orient='index', columns=sheet_names
    ).T
    return info, keyword_dict, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='A tool to generate visualizations of statistics related to a specific keyword from a local file'
    )
    parser.add_argument('--update-cache', action='store_true')
    args = parser.parse_args()
    paper_info, keywd_dict, paper_statistics = load_paper_info(args)

    app = Dash(__name__)

    columns_info = [
        {"name": str(i), "id": str(i)} for i in paper_info["CVPR_2019"].columns[:-1]
    ]
    columns_info[3]["presentation"] = "markdown"

    app.layout = html.Div(
        [
            html.H4('CV Conference Statistics', style={'fontSize': '36px'}),
            dcc.Graph(id="keyword-chart"),
            html.P("Select keyword (descending order):", style={'fontSize': '24px'}),
            dcc.Dropdown(
                id="ticker",
                options=[*keywd_dict.keys()],
                value="image",
                clearable=False,
                style={'fontSize': '24px'},
            ),
            dash_table.DataTable(
                id='table',
                columns=columns_info,
                data=[],
                markdown_options={"html": True},
                css=[
                    {
                        'selector': '.dash-spreadsheet td div',
                        'rule': '''
                            display: block;
                            overflow-y: hidden;
                            selector: p;
                            margin: 0;
                            text-align: center;
                        ''',
                    }
                ],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'backgroundColor': 'lavender',
                },
                style_table={
                    'overflowX': 'auto',
                    'height': '600px',
                    'overflowY': 'auto',
                },
                style_cell={
                    'fontSize': '24px',
                    'textAlign': 'center',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_header=dict(backgroundColor="paleturquoise"),
            ),
        ]
    )

    @app.callback(Output("keyword-chart", "figure"), Input("ticker", "value"))
    def display_time_series(keyword):
        fig = px.line(paper_statistics, x='Conference', y=keyword)
        fig.update_layout(
            font=dict(size=20, color="black"),
        )
        return fig

    @app.callback([Output("table", "data")], [Input("ticker", "value")])
    def updateTable(keyword):
        return generate_keywd_list(paper_info, keywd_dict[keyword][1])

    app.run_server(debug=False, port=8050, host='0.0.0.0')
