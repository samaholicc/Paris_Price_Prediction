from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

# Load dataset
dvf_agg = pd.read_csv('cleaned_paris_dvf_2024.csv')

# Figures
fig_hist = px.histogram(dvf_agg, x='valeur_fonciere', nbins=50,
                        title='Distribution des prix immobiliers à Paris (2024)',
                        labels={'valeur_fonciere': 'Prix de vente (€)', 'count': 'Fréquence'})
fig_box = px.box(dvf_agg, x='arrondissement', y='valeur_fonciere',
                 title='Prix immobiliers par arrondissement à Paris (2024)',
                 labels={'arrondissement': 'Arrondissement', 'valeur_fonciere': 'Prix de vente (€)'})
fig_scatter = px.scatter(dvf_agg, x='surface_reelle_bati', y='valeur_fonciere',
                         color='arrondissement', size='nombre_pieces_principales',
                         hover_data=['arrondissement'],
                         title='Surface habitable vs Prix de vente à Paris (2024)',
                         labels={'surface_reelle_bati': 'Surface habitable (m²)', 'valeur_fonciere': 'Prix de vente (€)'})
fig_map = px.scatter_mapbox(dvf_agg, lat='latitude', lon='longitude',
                            color='valeur_fonciere', size='surface_reelle_bati',
                            hover_data=['arrondissement', 'valeur_fonciere', 'price_per_m2'],
                            title='Prix immobiliers à Paris par localisation (2024)',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            zoom=10, color_continuous_midpoint=500000)
fig_map.update_layout(mapbox_style="open-street-map")

# Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Prédiction des prix immobiliers à Paris (2024)"),
    html.H3("Projet Big Data & IA pour alternance 2025"),
    dcc.Graph(figure=fig_hist),
    dcc.Graph(figure=fig_box),
    dcc.Graph(figure=fig_scatter),
    dcc.Graph(figure=fig_map)
])

if __name__ == '__main__':
    app.run(debug=True)