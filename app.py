import os
import pandas as pd
import dash
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# data updates 
df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
df['AnnualContractValue']=df['MonthlyCharges']*12

df['InternetType']=df['InternetService']
df['InternetService'] = df['InternetService'].map({'Fiber optic':'Yes','DSL':'Yes','No':'No'})

product_features = [
    'PhoneService', 'InternetService', 
    'OnlineBackup', 'OnlineSecurity', 
    'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies'
]

for feature in product_features:
    df[feature].replace({'Yes':1,'No':0},inplace=True)

# data subsets
tenure_churn = df[df['tenure']>0].groupby('tenure').Churn.mean().reset_index()

colors = {
    'background': '#FFFFFF',
    'text': 'black',
    'plotbg': "#DCDCDC"
}

app.layout = html.Div(children=[
    html.H1(
        children='Churn Predictions',
        style={
            'textAlign':'center',
            'color':colors['text']
        }
    ),
    
    html.Div(children='''Investigating how features affect customer retention''', style={
        'textAlign':'center',
        'color':colors['text']
    }),
    
    html.Div(
        [dcc.Markdown(children="""
    The following scatter graph shows average retention rates for customer groups based on tenure and contract type. 
    Hover over the markers to see the data ('tenure','average churn rate for the group') and the number of customers in the group.
    """)
        ]
    ),
    
    html.Div([
        html.Div([
            html.Label('Product Filter - show contracts that include all selected products'),
            dcc.Dropdown(
                id='product-filter',
                options=[{'label':i,'value':i} for i in product_features],
                value = [],
                multi=True
            )
        ],
        style={'width':'48%','display': 'inline-block'})
    ]),
    dcc.Graph(id='scatter-with-productFilter'),
    
    dcc.Graph(
        id='churn-bar-withSlider',
        figure={
            'data':[
                dict(
                    x=df.groupby('Contract')['Churn'].mean().reset_index()['Contract'],
                    y=df.groupby('Contract')['Churn'].mean().reset_index()['Churn'],
                    type='bar'
                )
            ],
            'layout': dict(title='Churn Rate by Contract Type')
        },
    ),
    
    html.H4(children='Customer Churn - The Data'),
    generate_table(df)
    
])

@app.callback(
    Output('scatter-with-productFilter','figure'),
    [Input('product-filter','value')]
)
def update_graph(product_filter_values):
    i = []
    for col in product_filter_values:
        i.extend(df[df[col]==1].index)
    dff = df.iloc[
        df.index.difference(pd.Index(i).unique())
    ]
    
    return {
        'data': [
            dict(
                x=dff[(dff['tenure']>0)&(dff['Contract'] == i)].groupby('tenure')['Churn'].mean().reset_index()['tenure'],
                y=dff[(dff['tenure']>0)&(dff['Contract'] == i)].groupby('tenure')['Churn'].mean().reset_index()['Churn'],
                text = dff[(dff['tenure']>0)&(df['Contract'] == i)].groupby('tenure')['Churn'].count(),
                mode = 'markers',
                opacity = 0.7,
                marker = {
                    'size':15,
                    'line': {'width': 0.5, 'color':'white'}
                 },
                 name = i 
             ) for i in df.Contract.unique()
         ],
         'layout': dict(
             xaxis = {'title':'Tenure (no. months)','dtick':12},
             yaxis = {'title': 'Average Churn'},
             margin = {'l': 40, 'b':40, 't': 10, 'r': 10},
             legend = {"x":0,"y":1},
             hovermode='closest'
         )
        }

if __name__ == '__main__':
    app.run_server(debug=True)