import json

from dash import Dash, html, dcc, callback, ctx, Input, Output
from get_metadata import get_meta_data

app = Dash()

app.layout = [
    html.Div(
        className="app-header",
        children=[
            html.Div('MLLO System', className="app-header-title")
        ]
    ),
    html.Div([
        html.H1('Keras Model to JSON'),
        dcc.Input(
            id='input-model-path',
            placeholder='Please type the model path.',
            type='text',
            value='toy_model'
        ),
        html.Button('Convert', id='btn-model-to-json', className="app-button", n_clicks=0),
        dcc.Loading([
            dcc.Textarea(
                id='textarea-json',
                value='',
                style={'width': '100%', 'height': 300},
            ),
        ], type="circle")
    ])
]


@callback(
    Output('textarea-json', 'value'),
    [Input("btn-model-to-json", "n_clicks"),
     Input("input-model-path", "value"), ]
)
def update_loading_div(n_clicks, value):
    if 'btn-model-to-json' == ctx.triggered_id:
        mllo_dict = get_meta_data(value, framework='KerasTensorFlow')
        return json.dumps(mllo_dict, indent=4)


if __name__ == '__main__':
    app.run(debug=True)
