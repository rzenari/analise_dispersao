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

@st.cache_data
def carregar_dados(arquivo_enviado):
    """Lê o arquivo de forma otimizada, com depuração de erros aprimorada."""
    colunas_necessarias = [
        'latitude', 'longitude', 'sucursal', 
        'centro_operativo', 'corte_recorte', 'prioridade'
    ]
    
    def processar_dataframe(df):
        """Função auxiliar para limpar e processar o dataframe."""
        df.columns = df.columns.str.lower().str.strip()
        
        if not all(col in df.columns for col in colunas_necessarias):
            st.error("ERRO: Colunas essenciais não foram encontradas após o carregamento.")
            st.write("Colunas necessárias:", colunas_necessarias)
            st.write("Colunas encontradas:", df.columns.tolist())
            return None

        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df

    # Tenta ler como CSV
    try:
        arquivo_enviado.seek(0)
        df_csv = pd.read_csv(
            arquivo_enviado, 
            encoding='utf-16', # Priorizando utf-16 que diagnosticamos antes
            sep='\t', # Arquivos utf-16 geralmente usam TAB como separador
            usecols=lambda column: column.strip().lower() in colunas_necessarias
        )
        st.success("Arquivo CSV lido com sucesso (codificação: utf-16, separador: TAB).")
        return processar_dataframe(df_csv)
    except Exception as e_utf16:
        # Se falhar, agora vamos mostrar o erro para depuração
        st.warning(f"Falha ao ler como CSV (UTF-16/TAB). Tentando outros formatos... (Erro: {e_utf16})")

        # Tenta outros formatos de CSV
        try:
            arquivo_enviado.seek(0)
            df_csv_latin = pd.read_csv(
                arquivo_enviado, 
                encoding='latin-1',
                sep=None,
                engine='python',
                usecols=lambda column: column.strip().lower() in colunas_necessarias
            )
            st.success("Arquivo CSV lido com sucesso (codificação: latin-1).")
            return processar_dataframe(df_csv_latin)
        except Exception as e_latin:
            st.warning(f"Falha ao ler como CSV (Latin-1). Tentando Excel... (Erro: {e_latin})")

    # Tenta ler como Excel (agora também otimizado)
    try:
        arquivo_enviado.seek(0)
        # Otimização: primeiro lemos o cabeçalho para saber quais colunas carregar
        df_header = pd.read_excel(arquivo_enviado, engine='openpyxl', nrows=0)
        df_header.columns = df_header.columns.str.lower().str.strip()
        
        colunas_para_carregar = [col for col in df_header.columns if col in colunas_necessarias]
        
        df_excel = pd.read_excel(
            arquivo_enviado, 
            engine='openpyxl',
            usecols=colunas_para_carregar # Carrega apenas as colunas necessárias
        )
        st.success("Arquivo lido com sucesso como Excel (.xlsx) de forma otimizada.")
        return processar_dataframe(df_excel)
    except Exception as e_excel:
        st.error(f"Não foi possível ler o arquivo. O último erro foi na tentativa de ler como Excel: {e_excel}")
        return None

# ... (o resto do código permanece exatamente o mesmo) ...

def calcular_nni(gdf):
    """Calcula o Índice do Vizinho Mais Próximo (NNI)."""
    if len(gdf) < 3:
        return None, "Pontos insuficientes (< 3)."

    n_points = len(gdf)
    points = np.array([gdf.geometry.x, gdf.geometry.y]).T
    
    try:
        total_bounds = gdf.total_bounds
        area = (total_bounds[2] - total_bounds[0]) * (total_bounds[3] - total_bounds[1])
        if area == 0:
            return None, "Área inválida (pontos colineares)."

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
    if gdf.empty or len(gdf) < min_samples:
        gdf['cluster'] = -1
        return gdf
    
    raio_terra_km = 6371
    eps_rad = eps_km / raio_terra_km
    
    coords = np.radians(gdf[['latitude', 'longitude']].values)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    gdf['cluster'] = db.labels_
    return gdf

st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader(
    "Escolha a planilha de cortes",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    df_completo = carregar_dados(uploaded_file)

    if df_completo is not None:
        st.sidebar.success(f"{len(df_completo)} registros carregados!")
        
        st.sidebar.markdown("### Filtros da Análise")
        
        filtros = ['sucursal', 'centro_operativo', 'corte_recorte', 'prioridade']
        valores_selecionados = {}

        for coluna in filtros:
            if coluna in df_completo.columns:
                opcoes = ["Todos"] + sorted(df_completo[coluna].dropna().unique().tolist())
                valores_selecionados[coluna] = st.sidebar.selectbox(f"{coluna.replace('_', ' ').title()}", opcoes)

        df_filtrado = df_completo.copy()
        for coluna, valor in valores_selecionados.items():
            if coluna in df_filtrado.columns and valor != "Todos":
                df_filtrado = df_filtrado[df_filtrado[coluna] == valor]

        st.header("Resultados da Análise")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Cortes Carregados", len(df_completo))
        col2.metric("Cortes na Seleção Atual", len(df_filtrado))
        
        if not df_filtrado.empty:
            gdf_filtrado = gpd.GeoDataFrame(
                df_filtrado,
                geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude),
                crs="EPSG:4326"
            )
            
            nni_valor, nni_texto = calcular_nni(gdf_filtrado)
            col3.metric("Padrão de Dispersão", nni_texto)
            
            st.sidebar.markdown("### Parâmetros de Cluster")
            eps_cluster_km = st.sidebar.slider("Raio do Cluster (km)", 0.1, 5.0, 1.0, 0.1)
            min_samples_cluster = st.sidebar.slider("Mínimo de Pontos por Cluster", 2, 20, 5, 1)

            gdf_com_clusters = executar_dbscan(gdf_filtrado, eps_km=eps_cluster_km, min_samples=min_samples_cluster)
            
            n_clusters = len(set(gdf_com_clusters['cluster'])) - (1 if -1 in gdf_com_clusters['cluster'] else 0)
            n_ruido = list(gdf_com_clusters['cluster']).count(-1)

            st.subheader(f"Análise de Clusters: {n_clusters} hotspots encontrados")
            st.write(f"Foram encontrados **{n_clusters} clusters** e **{n_ruido} pontos isolados** com os parâmetros definidos.")
            
            if not gdf_com_clusters.empty:
                map_center = [gdf_com_clusters.latitude.mean(), gdf_com_clusters.longitude.mean()]
                m = folium.Map(location=map_center, zoom_start=11)

                unique_clusters = sorted(gdf_com_clusters['cluster'].unique())
                colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(unique_clusters))]
                cluster_colors = {cluster_id: color for cluster_id, color in zip(unique_clusters, colors)}
                
                for idx, row in gdf_com_clusters.iterrows():
                    cluster_id = row['cluster']
                    color = cluster_colors.get(cluster_id, '#000000') if cluster_id != -1 else '#000000'
                    
                    popup_text = ""
                    for col in ['prioridade', 'centro_operativo', 'corte_recorte']:
                        if col in row:
                            popup_text += f"{col.replace('_', ' ').title()}: {row[col]}<br>"

                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=popup_text
                    ).add_to(m)
                
                st_folium(m, width=725, height=500, returned_objects=[])
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")

else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")
