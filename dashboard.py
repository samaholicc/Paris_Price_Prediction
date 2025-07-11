from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# Load dataset from GitHub raw URL
dvf_agg = pd.read_csv('https://raw.githubusercontent.com/samaholicc/Paris_Price_Prediction/master/cleaned_paris_dvf_2024.csv')

# Train LightGBM model
X = dvf_agg[['surface_reelle_bati', 'nombre_pieces_principales', 'arrondissement', 'distance_to_center']]
y = dvf_agg['valeur_fonciere']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Recommendation function
def recommend_properties(surface, rooms, arrondissement, budget, top_n=5):
    filtered_df = dvf_agg[
        (dvf_agg['surface_reelle_bati'].between(surface-10, surface+10)) &
        (dvf_agg['nombre_pieces_principales'].between(rooms-1, rooms+1)) &
        (dvf_agg['arrondissement'] == arrondissement)
    ]
    if filtered_df.empty:
        return pd.DataFrame()
    filtered_df['price_diff'] = abs(filtered_df['valeur_fonciere'] - budget)
    return filtered_df.sort_values('price_diff').head(top_n)[['surface_reelle_bati', 'nombre_pieces_principales', 'arrondissement', 'valeur_fonciere', 'price_per_m2']]

# Figures with Parisian-themed styling
fig_hist = px.histogram(dvf_agg, x='valeur_fonciere', nbins=50,
                        title='Distribution des prix immobiliers à Paris (2024)',
                        labels={'valeur_fonciere': 'Prix de vente (€)', 'count': 'Fréquence'},
                        color_discrete_sequence=['#1E3A8A'])
fig_box = px.box(dvf_agg, x='arrondissement', y='valeur_fonciere',
                 title='Prix immobiliers par arrondissement à Paris (2024)',
                 labels={'arrondissement': 'Arrondissement', 'valeur_fonciere': 'Prix de vente (€)'},
                 color_discrete_sequence=['#1E3A8A'])
fig_scatter = px.scatter(dvf_agg, x='surface_reelle_bati', y='valeur_fonciere',
                         color='arrondissement', size='nombre_pieces_principales',
                         hover_data=['arrondissement'],
                         title='Surface habitable vs Prix de vente à Paris (2024)',
                         labels={'surface_reelle_bati': 'Surface habitable (m²)', 'valeur_fonciere': 'Prix de vente (€)'},
                         color_continuous_scale='Viridis')
fig_map = px.scatter_mapbox(dvf_agg, lat='latitude', lon='longitude',
                            color='valeur_fonciere', size='surface_reelle_bati',
                            hover_data=['arrondissement', 'valeur_fonciere', 'price_per_m2'],
                            title='Prix immobiliers à Paris par localisation (2024)',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            zoom=10, color_continuous_midpoint=500000)
fig_map.update_layout(mapbox_style="open-street-map", title_font=dict(size=20, color='#D4A017'))

# Animated background CSS
animated_background = """
<style>
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.animated-bg {
    background: linear-gradient(270deg, #1E3A8A, #4B5EAA, #1E3A8A);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
</style>
"""

# Dash app with Tailwind CSS and animated background
app = Dash(__name__, external_stylesheets=['https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'])
app.index_string = animated_background + app.index_string
app.layout = html.Div(className='animated-bg text-white min-h-screen p-8', children=[
    html.H1("Prédiction des Prix Immobiliers à Paris", className='text-5xl font-bold text-center mb-8 text-yellow-400'),
    html.Div(className='bg-white bg-opacity-90 text-black p-6 rounded-lg shadow-2xl mb-8 max-w-3xl mx-auto', children=[
        html.Label("Filtrer par arrondissement", className='text-lg font-semibold text-blue-900 mb-2 block'),
        dcc.Dropdown(
            id='arrondissement-filter',
            options=[{'label': f'Arrondissement {i}', 'value': i} for i in sorted(dvf_agg['arrondissement'].unique())],
            value=None,
            placeholder="Choisir un arrondissement",
            multi=True,
            className='w-full p-2 border rounded border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500'
        ),
        html.Label("Filtrer par type de local", className='text-lg font-semibold text-blue-900 mt-4 mb-2 block'),
        dcc.Dropdown(
            id='type-local-filter',
            options=[{'label': t, 'value': t} for t in dvf_agg['type_local'].unique()],
            value=None,
            placeholder="Choisir un type de local",
            className='w-full p-2 border rounded border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500'
        )
    ]),
    dcc.Graph(id='histogram', figure=fig_hist, className='bg-white bg-opacity-90 p-6 rounded-lg shadow-2xl mb-8'),
    dcc.Graph(id='boxplot', figure=fig_box, className='bg-white bg-opacity-90 p-6 rounded-lg shadow-2xl mb-8'),
    dcc.Graph(id='scatter', figure=fig_scatter, className='bg-white bg-opacity-90 p-6 rounded-lg shadow-2xl mb-8'),
    dcc.Graph(id='map', figure=fig_map, className='bg-white bg-opacity-90 p-6 rounded-lg shadow-2xl mb-8'),
    html.Div(className='bg-white bg-opacity-90 text-black p-6 rounded-lg shadow-2xl max-w-3xl mx-auto', children=[
        html.H3("Prédire le prix d’un appartement", className='text-2xl font-semibold text-blue-900 mb-4'),
        html.Label("Surface habitable (m²)", className='block mb-2 text-blue-900'),
        dcc.Input(id='surface-input', type='number', value=50, className='w-full p-2 border rounded border-blue-300 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500'),
        html.Label("Nombre de pièces", className='block mb-2 text-blue-900'),
        dcc.Input(id='rooms-input', type='number', value=2, className='w-full p-2 border rounded border-blue-300 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500'),
        html.Label("Arrondissement", className='block mb-2 text-blue-900'),
        dcc.Dropdown(
            id='arrondissement-input',
            options=[{'label': f'Arrondissement {i}', 'value': i} for i in sorted(dvf_agg['arrondissement'].unique())],
            value=1,
            className='w-full p-2 border rounded border-blue-300 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500'
        ),
        html.Label("Distance au centre (km)", className='block mb-2 text-blue-900'),
        dcc.Input(id='distance-input', type='number', value=2, className='w-full p-2 border rounded border-blue-300 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500'),
        html.Label("Budget maximum (€)", className='block mb-2 text-blue-900'),
        dcc.Input(id='budget-input', type='number', value=500000, className='w-full p-2 border rounded border-blue-300 mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500'),
        html.Button('Prédire & Recommander', id='predict-button', n_clicks=0, className='bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-700 transition duration-300 w-full'),
        html.Div(id='prediction-output', className='mt-4 text-lg font-semibold text-blue-900'),
        html.Div(id='recommendation-output', className='mt-4 text-lg font-semibold text-blue-900')
    ])
])

@app.callback(
    [Output('histogram', 'figure'),
     Output('boxplot', 'figure'),
     Output('scatter', 'figure'),
     Output('map', 'figure')],
    [Input('arrondissement-filter', 'value'),
     Input('type-local-filter', 'value')]
)
def update_graphs(arrondissements, type_local):
    filtered_df = dvf_agg
    if arrondissements:
        filtered_df = filtered_df[filtered_df['arrondissement'].isin(arrondissements)]
    if type_local:
        filtered_df = filtered_df[filtered_df['type_local'] == type_local]
    fig_hist = px.histogram(filtered_df, x='valeur_fonciere', nbins=50,
                            title='Distribution des prix immobiliers à Paris (2024)',
                            labels={'valeur_fonciere': 'Prix de vente (€)', 'count': 'Fréquence'},
                            color_discrete_sequence=['#1E3A8A'])
    fig_box = px.box(filtered_df, x='arrondissement', y='valeur_fonciere',
                     title='Prix immobiliers par arrondissement à Paris (2024)',
                     labels={'arrondissement': 'Arrondissement', 'valeur_fonciere': 'Prix de vente (€)'},
                     color_discrete_sequence=['#1E3A8A'])
    fig_scatter = px.scatter(filtered_df, x='surface_reelle_bati', y='valeur_fonciere',
                             color='arrondissement', size='nombre_pieces_principales',
                             hover_data=['arrondissement'],
                             title='Surface habitable vs Prix de vente à Paris (2024)',
                             labels={'surface_reelle_bati': 'Surface habitable (m²)', 'valeur_fonciere': 'Prix de vente (€)'},
                             color_continuous_scale='Viridis')
    fig_map = px.scatter_mapbox(filtered_df, lat='latitude', lon='longitude',
                                color='valeur_fonciere', size='surface_reelle_bati',
                                hover_data=['arrondissement', 'valeur_fonciere', 'price_per_m2'],
                                title='Prix immobiliers à Paris par localisation (2024)',
                                color_continuous_scale=px.colors.sequential.Plasma,
                                zoom=10, color_continuous_midpoint=500000)
    fig_map.update_layout(mapbox_style="open-street-map", title_font=dict(size=20, color='#D4A017'))
    return fig_hist, fig_box, fig_scatter, fig_map

@app.callback(
    [Output('prediction-output', 'children'),
     Output('recommendation-output', 'children')],
    [Input('predict-button', 'n_clicks')],
    [Input('surface-input', 'value'),
     Input('rooms-input', 'value'),
     Input('arrondissement-input', 'value'),
     Input('distance-input', 'value'),
     Input('budget-input', 'value')]
)
def predict_and_recommend(n_clicks, surface, rooms, arrondissement, distance, budget):
    if n_clicks > 0 and all([surface, rooms, arrondissement, distance, budget]):
        input_data = pd.DataFrame([[surface, rooms, arrondissement, distance]],
                                  columns=['surface_reelle_bati', 'nombre_pieces_principales', 'arrondissement', 'distance_to_center'])
        prediction = model.predict(input_data)[0]
        recommendations = recommend_properties(surface, rooms, arrondissement, budget)
        recommendation_text = "Propriétés recommandées :\n" + recommendations.to_string(index=False) if not recommendations.empty else "Aucune propriété trouvée dans ces critères."
        return f'Prix prédit (LightGBM) : {prediction:.2f} €', recommendation_text
    return "Entrez les valeurs et cliquez sur Prédire & Recommander.", ""

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)