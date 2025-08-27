# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from math import sqrt
import folium
from streamlit_folium import st_folium

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E TÍTULOS
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análise de Dispersão Geográfica")

st.title("🗺️ Ferramenta de Análise de Dispersão Geográfica")
st.write("""
    Faça o upload da sua planilha de cortes para analisar a distribuição geográfica,
    identificar clusters e obter insights para sua operação logística.
""")

# ==============================================================================
# 3. FUNÇÕES DE ANÁLISE (COM CACHE PARA PERFORMANCE)
# ==============================================================================

# O cache do Streamlit guarda o resultado da função. Se a função for chamada
# com os mesmos parâmetros, ele retorna o resultado guardado sem reexecutar.
@st.cache_data
def carregar_dados(arquivo_enviado):
    """Lê o arquivo CSV ou Excel, limpa e prepara os dados."""
    try:
        df = pd.read_csv(arquivo_enviado)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {e}")
        st.info("Tentando ler como arquivo Excel...")
        try:
            df = pd.read_excel(arquivo_enviado)
        except Exception as e2:
            st.error(f"Erro ao ler o arquivo Excel: {e2}")
            return None

    # Validação de colunas essenciais
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("O arquivo precisa conter as colunas 'latitude' e 'longitude'.")
        return None

    # Limpeza dos dados de coordenadas (substitui vírgula por ponto)
    df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
    df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
    df = df.dropna(subset=['latitude', 'longitude'])
    return df

def calcular_nni(gdf):
    """Calcula o Índice do Vizinho Mais Próximo (NNI)."""
    if len(gdf) < 3:
        return None, "Pontos insuficientes (< 3) para calcular NNI."

    n_points = len(gdf)
    points = np.array([gdf.geometry.x, gdf.geometry.y]).T
    
    try:
        total_bounds = gdf.total_bounds
        area = (total_bounds[2] - total_bounds[0]) * (total_bounds[3] - total_bounds[1])
        if area == 0:
            return None, "Área inválida (todos os pontos estão na mesma linha)."

        dist_matrix = distance.cdist(points, points, 'euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        
        nearest_neighbor_dists = dist_matrix.min(axis=1)
        observed_mean_dist = nearest_neighbor_dists.mean()
        
        expected_mean_dist = 0.5 * sqrt(area / n_points)
        
        nni = observed_mean_dist / expected_mean_dist
        
        if nni < 1:
            interpretacao = f"Agrupado (NNI: {nni:.2f})"
        elif nni > 1:
            interpretacao = f"Disperso (NNI: {nni:.2f})"
        else:
            interpretacao = f"Aleatório (NNI: {nni:.2f})"
        
        return nni, interpretacao
    except Exception as e:
        return None, f"Erro no cálculo: {e}"

def executar_dbscan(gdf, eps_km=0.5, min_samples=3):
    """Executa o DBSCAN para encontrar clusters."""
    if gdf.empty:
        return gdf
    
    # Conversão de eps de km para radianos (aproximação)
    raio_terra_km = 6371
    eps_rad = eps_km / raio_terra_km
    
    coords = np.radians(gdf[['latitude', 'longitude']].values)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    gdf['cluster'] = db.labels_
    return gdf

# ==============================================================================
# 4. BARRA LATERAL (SIDEBAR) COM UPLOAD E FILTROS
# ==============================================================================
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader(
    "Escolha a planilha de cortes",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    df_completo = carregar_dados(uploaded_file)

    if df_completo is not None:
        st.sidebar.success(f"{len(df_completo)} registros carregados com sucesso!")
        
        # --- Geração Dinâmica dos Filtros ---
        st.sidebar.markdown("### Filtros da Análise")
        
        # Filtro de Sucursal
        sucursais = ["Todas"] + sorted(df_completo['sucursal'].unique().tolist())
        sucursal_selecionada = st.sidebar.selectbox("Sucursal", sucursais)

        # Filtro de Centro Operativo
        centros_op = ["Todos"] + sorted(df_completo['Centro_Operativo'].unique().tolist())
        co_selecionado = st.sidebar.selectbox("Centro Operativo", centros_op)

        # Filtro de Corte/Recorte
        cortes_recortes = ["Todos"] + sorted(df_completo['corte_recorte'].unique().tolist())
        cr_selecionado = st.sidebar.selectbox("Corte/Recorte", cortes_recortes)
        
        # Filtro de Prioridade
        prioridades = ["Todas"] + sorted(df_completo['Prioridade'].unique().tolist())
        prioridade_selecionada = st.sidebar.selectbox("Prioridade", prioridades)

        # --- Aplicação dos Filtros ---
        df_filtrado = df_completo.copy()
        if sucursal_selecionada != "Todas":
            df_filtrado = df_filtrado[df_filtrado['sucursal'] == sucursal_selecionada]
        if co_selecionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['Centro_Operativo'] == co_selecionado]
        if cr_selecionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['corte_recorte'] == cr_selecionado]
        if prioridade_selecionada != "Todas":
            df_filtrado = df_filtrado[df_filtrado['Prioridade'] == prioridade_selecionada]

        # ==============================================================================
        # 5. ÁREA PRINCIPAL COM RESULTADOS E MAPA
        # ==============================================================================
        
        st.header("Resultados da Análise")
        
        # --- Métricas ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Cortes no Arquivo", len(df_completo))
        col2.metric("Cortes na Seleção Atual", len(df_filtrado))
        
        # Criar GeoDataFrame para as análises
        gdf_filtrado = gpd.GeoDataFrame(
            df_filtrado,
            geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude),
            crs="EPSG:4326"
        )
        
        # --- Análise de Dispersão (NNI) ---
        nni_valor, nni_texto = calcular_nni(gdf_filtrado)
        col3.metric("Padrão de Dispersão", nni_texto)
        
        # --- Análise de Cluster (DBSCAN) ---
        st.sidebar.markdown("### Parâmetros de Cluster")
        eps_cluster_km = st.sidebar.slider("Distância máx. para cluster (km)", 0.1, 5.0, 1.0, 0.1)
        min_samples_cluster = st.sidebar.slider("Nº min. de pontos por cluster", 2, 20, 5, 1)

        gdf_com_clusters = executar_dbscan(gdf_filtrado, eps_km=eps_cluster_km, min_samples=min_samples_cluster)
        
        n_clusters = len(set(gdf_com_clusters['cluster'])) - (1 if -1 in gdf_com_clusters['cluster'] else 0)
        n_ruido = list(gdf_com_clusters['cluster']).count(-1)

        st.subheader(f"Análise de Clusters: {n_clusters} hotspots encontrados")
        st.write(f"Foram encontrados **{n_clusters} clusters (hotspots)** e **{n_ruido} pontos isolados** com os parâmetros definidos.")
        
        # --- Mapa Interativo ---
        if not gdf_com_clusters.empty:
            map_center = [gdf_com_clusters.latitude.mean(), gdf_com_clusters.longitude.mean()]
            m = folium.Map(location=map_center, zoom_start=11)

            # Cores para os clusters
            unique_clusters = sorted(gdf_com_clusters['cluster'].unique())
            colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(unique_clusters))]
            cluster_colors = {cluster_id: color for cluster_id, color in zip(unique_clusters, colors)}
            
            for idx, row in gdf_com_clusters.iterrows():
                cluster_id = row['cluster']
                color = cluster_colors[cluster_id] if cluster_id != -1 else '#000000'
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=f"Cluster: {cluster_id}<br>Prioridade: {row['Prioridade']}"
                ).add_to(m)
            
            # Exibe o mapa no Streamlit
            st_folium(m, width=725, height=500, returned_objects=[])
        else:
            st.warning("Nenhum dado para exibir no mapa com os filtros atuais.")

else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")