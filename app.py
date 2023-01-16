from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pytchat
import json
import pythainlp.util as util
import pythainlp.tokenize as tokenize
import numpy as np
import dash_bootstrap_components as dbc
from pythainlp.corpus.common import thai_stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]

app = Dash(__name__, external_stylesheets=external_stylesheets)

vedioMessage = []
keyword = []
cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
lr = LogisticRegression()
dfOfTable = pd.DataFrame()
def generateModel():
    global thai_stopwords

    with open("kaggle-competition/train.txt", encoding="utf8") as f:
        texts = [line.strip() for line in f.readlines()]
    
    with open("kaggle-competition/train_label.txt", encoding="utf8") as f:
        categories = [line.strip() for line in f.readlines()]

    all_df = pd.DataFrame({"category":categories, "texts":texts})
    all_df.to_csv('kaggle-competition/all_df.csv',index=False)
    
    thai_stopwords = list(thai_stopwords())

    all_df['text_tokens'] = all_df['texts'].apply(text_process)
    X = all_df[['text_tokens']]
    Y = all_df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

    global cvec
    global lr

    cvec.fit_transform(X_train['text_tokens'])
    train_bow = cvec.transform(X_train['text_tokens'])
    lr.fit(train_bow, y_train)

def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = tokenize.word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() if word.lower not in thai_stopwords)
    return final
    

def loadVedio(value):
    try:
        chat = pytchat.create(video_id=value) #wRaGJufvhfs
        item = []
        while chat.is_alive():
            result = json.loads(chat.get().json())
            for c in result:
                message = util.normalize(c['message'])
                elapsedTime = c['elapsedTime']
                global vedioMessage
                vedioMessage.append(message)

    except Exception as e: print(f"[ERROR]: {e}")

generateModel()
loadVedio("wRaGJufvhfs")

app.layout = html.Div(children=[
    html.Div(
        className = "centent-wrapper",
        children = [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            className = "box-centent-wrapper",
                            children=[
                                html.H1("คำที่ใช้บ่อย", style = { "flex": "1" }),
                                html.H1("จำแนกความรู้สึก", style = { "flex": "1" })
                            ],
                            style = { "height": "auto" }
                        ),
                        html.Div(
                            className = "box-centent-wrapper",
                            children=[
                                dcc.Graph(
                                    id='top-data',
                                    style = { "flex": "1" }
                                ),
                                dcc.Graph(
                                    id='message-data',
                                    style = { "flex": "1" }
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H1("เลือกจำนวนคำที่ใช้บ่อย", style = { "flex": "1" }),
                                dcc.Slider(
                                    1,
                                    10,
                                    step=1,
                                    id='select-top-slider',
                                    value=1,
                                )
                            ], style={'width': '49%', 'padding': '0px 20px 20px 20px'})
                    ]
                )
            )
        ]
    ),
    html.Div(
        className = "centent-wrapper",
        children = [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H1("ตัวอย่างข้อความ", style = { "flex": "1" }),
                        html.Div(
                            className = "box-centent-wrapper",
                            children=[
                                html.Table(id='tbl')
                            ],
                            style={'fontSize': '14px', 'height': 'auto'}
                        ),
                    ]
                )
            )
        ]
    ),
], style = { "padding": "1rem", "backgroundColor": "whitesmoke",  "height": "100vh"})

@app.callback(
    Output('top-data', 'figure'),
    Input('select-top-slider', 'value'))
def update_graph(topValue):
    try:
        horizontal = make_subplots(rows=1, cols=1, specs=[[{}]], shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001)
        global keyword, vedioMessage
        item = []
        if len(keyword) == 0 and len(vedioMessage) > 0:
            for message in vedioMessage:
                if util.countthai(message) >= 40:
                    text = tokenize.word_tokenize(message, engine="newmm", keep_whitespace=False)
                    item.extend(util.maiyamok(filter(lambda x : util.countthai(x) >= 50, text)))
            keyword = util.find_keyword(item,  min_len=1)    

        if len(keyword) > 0:
            df = pd.DataFrame(list(zip(list(keyword.keys()), list(keyword.values()))), columns =['name', 'val'])
            df = df.sort_values(by='val', ascending=False).head(topValue)
            
            horizontal.append_trace(go.Bar(
                x=df["val"],
                y=df["name"],
                marker=dict(
                    color='rgba(50, 171, 96, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 96, 1.0)',
                        width=1
                    ),
                ),
                orientation='h',
            ), 1, 1)

            horizontal.update_layout(
                yaxis=dict(
                    showgrid=False,
                    showline=False,
                    showticklabels=True,
                    domain=[0, 0.85],
                ),
                xaxis=dict(
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    showgrid=False,
                    domain=[0, 0.42],
                ),
                legend=dict(x=0.029, y=1.038, font_size=10),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgb(255,255,255)',
                plot_bgcolor='rgb(255,255,255)',
            )

            horizontalAnnotations = []

            y_s = np.round(df["val"], decimals=2)

            for yd, xd in zip(y_s, df["name"]):
                horizontalAnnotations.append(dict(xref='x1', yref='y1',
                                y=xd, x=yd + 300,
                                text=str(yd) + ' คำ',
                                font=dict(family='Arial', size=12,
                                        color='rgb(50, 171, 96)'),
                                showarrow=False))
            horizontal.update_layout(annotations=horizontalAnnotations)
    except Exception as e: print(f"[ERROR]: {e}")
    return horizontal

@app.callback(
    Output('message-data', 'figure'),
    Input('top-data', 'clickData'))
def update_graph2(clickData):
    try:
        global vedioMessage
        global dfOfTable

        horizontal = make_subplots(rows=1, cols=1, specs=[[{}]], shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001)
        label = str(clickData["points"][0]["label"])
        data = list(filter(lambda x : label in x, vedioMessage))
        value = list(map(getResult, data))
   
        df = pd.DataFrame(list(zip(data, value)), columns = ['message', 'val'])
        dfOfTable = df
        df_pos = df[df['val'] == 'pos']
        df_neg = df[df['val'] == 'neg']
        df_neu = df[df['val'] == 'neu']

        df_sentiment = pd.DataFrame(list(zip(list(['pos', 'neg', 'neu']), list([len(df_pos), len(df_neg), len(df_neu)]))), columns =['name', 'val'])
      
        horizontal.append_trace(go.Bar(
                x=df_sentiment["val"],
                y=df_sentiment["name"],
                marker=dict(
                    color='rgba(50, 171, 96, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 96, 1.0)',
                        width=1
                    ),
                ),
                orientation='h',
        ), 1, 1)

        horizontal.update_layout(
                yaxis=dict(
                    showgrid=False,
                    showline=False,
                    showticklabels=True,
                    domain=[0, 0.85],
                ),
                xaxis=dict(
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    showgrid=False,
                    domain=[0, 0.42],
                ),
                legend=dict(x=0.029, y=1.038, font_size=10),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgb(255,255,255)',
                plot_bgcolor='rgb(255,255,255)',
            )

        horizontalAnnotations = []

        y_s = np.round(df_sentiment["val"], decimals=2)

        for yd, xd in zip(y_s, df_sentiment["name"]):
            horizontalAnnotations.append(dict(xref='x1', yref='y1',
                y=xd, x=yd + 300,
                text=str(yd) + ' ข้อความ',
                font=dict(family='Arial', size=12,
                color='rgb(50, 171, 96)'),
                showarrow=False))
        horizontal.update_layout(annotations=horizontalAnnotations)
    except Exception as e: print(f"[ERROR]: {e}")
    return horizontal

@app.callback(
    Output('tbl', 'children'),
    Input('message-data', 'clickData'))
def update_table(clickData):
    global dfOfTable
    df = pd.DataFrame()
    max_rows=10
    try:
        label = str(clickData["points"][0]["label"])
        print(label)
        df = dfOfTable
        df = df[df['val'] == label]
        
    except Exception as e: print(f"[ERROR]: {e}")
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in df.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(min(len(df), max_rows))
        ])
    ])

def getResult(my_text):
    global lr
    my_tokens = text_process(my_text)
    my_bow = cvec.transform(pd.Series([my_tokens]))
    my_predictions = lr.predict(my_bow)
    return my_predictions[0]
    
if __name__ == '__main__':
    app.run_server(debug=True)