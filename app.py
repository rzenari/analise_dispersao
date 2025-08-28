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
from folium.plugins import MarkerCluster, HeatMap

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E TÍTULOS
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análise de Dispersão Geográfica")

st.title("🗺️ Ferramenta de Análise de Dispersão Geográfica")
st.write("Faça o upload da sua planilha de cortes para analisar a distribuição geográfica e identificar clusters")

# ==============================================================================
# 3. FUNÇÕES DE ANÁLISE (COM CACHE PARA PERFORMANCE)
# ==============================================================================

@st.cache_data
def carregar_dados(arquivo_enviado):
    """Lê o arquivo de forma otimizada, carregando apenas as colunas necessárias."""
    colunas_necessarias = ['latitude', 'longitude', 'sucursal', 'centro_operativo', 'corte_recorte', 'prioridade']
    
    def processar_dataframe(df):
        df.columns = df.columns.str.lower().str.strip()
        if not all(col in df.columns for col in colunas_necessarias):
            st.error("ERRO: Colunas essenciais não foram encontradas."); st.write("Colunas necessárias:", colunas_necessarias); st.write("Colunas encontradas:", df.columns.tolist())
            return None
        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df

    try:
        arquivo_enviado.seek(0)
        df_csv = pd.read_csv(arquivo_enviado, encoding='utf-16', sep='\t', usecols=lambda c: c.strip().lower() in colunas_necessarias)
        st.success("Arquivo CSV lido com sucesso (codificação: utf-16)."); return processar_dataframe(df_csv)
    except Exception:
        try:
            arquivo_enviado.seek(0)
            df_csv_latin = pd.read_csv(arquivo_enviado, encoding='latin-1', sep=None, engine='python', usecols=lambda c: c.strip().lower() in colunas_necessarias)
            st.success("Arquivo CSV lido com sucesso (codificação: latin-1)."); return processar_dataframe(df_csv_latin)
        except Exception:
            try:
                arquivo_enviado.seek(0)
                df_excel = pd.read_excel(arquivo_enviado, engine='openpyxl')
                df_excel.columns = df_excel.columns.str.lower().str.strip()
                colunas_para_usar = [col for col in colunas_necessarias if col in df_excel.columns]
                df = df_excel[colunas_para_usar]
                st.success("Arquivo Excel lido com sucesso."); return processar_dataframe(df)
            except Exception as e:
                st.error(f"Não foi possível ler o arquivo. Último erro: {e}"); return None

def calcular_nni_otimizado(gdf):
    """Calcula NNI de forma otimizada para memória."""
    if len(gdf) < 3: return None, "Pontos insuficientes (< 3)."
    n_points = len(gdf)
    points = np.array([gdf.geometry.x, gdf.geometry.y]).T
    soma_dist_minimas = sum(distance.cdist([points[i]], np.delete(points, i, axis=0)).min() for i in range(n_points))
    observed_mean_dist = soma_dist_minimas / n_points
    try:
        total_bounds = gdf.total_bounds; area = (total_bounds[2] - total_bounds[0]) * (total_bounds[3] - total_bounds[1])
        if area == 0: return None, "Área inválida."
        expected_mean_dist = 0.5 * sqrt(area / n_points)
        nni = observed_mean_dist / expected_mean_dist
        if nni < 1: interpretacao = f"Agrupado (NNI: {nni:.2f})"
        elif nni > 1: interpretacao = f"Disperso (NNI: {nni:.2f})"
        else: interpretacao = f"Aleatório (NNI: {nni:.2f})"
        return nni, interpretacao
    except Exception as e: return None, f"Erro no cálculo: {e}"

def executar_dbscan(gdf, eps_km=0.5, min_samples=3):
    """Executa o DBSCAN para encontrar clusters."""
    if gdf.empty or len(gdf) < min_samples: gdf['cluster'] = -1; return gdf
    raio_terra_km = 6371; eps_rad = eps_km / raio_terra_km
    coords = np.radians(gdf[['latitude', 'longitude']].values)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    gdf['cluster'] = db.labels_
    return gdf

def gerar_resumo_didatico(nni_valor, n_clusters, percent_dispersos, is_media=False):
    """Gera um texto interpretativo considerando tanto o NNI quanto a % de dispersão."""
    if nni_valor is None: return ""
    
    prefixo = "Na média, o padrão" if is_media else "O padrão"

    if percent_dispersos > 50:
        titulo = "⚠️ **Padrão Misto (Agrupamentos Isolados)**"
        obs = f"Apesar da existência de **{n_clusters} hotspots**, a maioria dos serviços (**{percent_dispersos:.1f}%**) está **dispersa** pela região."
        acao = f"**Ação Recomendada:** Trate a operação de forma híbrida. Otimize rotas para os hotspots e agrupe os serviços dispersos por setor ou dia para aumentar a eficiência."
    elif nni_valor < 0.5:
        titulo = "📈 **Padrão Fortemente Agrupado (Excelente Oportunidade Logística)**"
        obs = f"{prefixo} dos cortes é **fortemente concentrado** em áreas específicas, com poucos serviços isolados."
        acao = f"**Ação Recomendada:** Crie rotas otimizadas para atender múltiplos chamados com baixo deslocamento. Avalie alocar equipes dedicadas para os **{n_clusters} hotspots** encontrados."
    elif 0.5 <= nni_valor < 0.8:
        titulo = "📊 **Padrão Moderadamente Agrupado (Potencial de Otimização)**"
        obs = f"{prefixo} dos cortes apresenta **boa concentração**, indicando a formação de clusters."
        acao = f"**Ação Recomendada:** Identifique os **{n_clusters} hotspots** mais densos para priorizar o roteamento. Há um bom potencial para agrupar serviços e reduzir custos."
    elif 0.8 <= nni_valor <= 1.2:
        titulo = "😐 **Padrão Aleatório (Sem Padrão Claro)**"
        obs = f"{prefixo} dos cortes é **aleatório**, sem concentração ou dispersão estatisticamente relevante."
        acao = f"**Ação Recomendada:** A logística para estes cortes tende a ser menos previsível. Considere uma abordagem de roteirização diária e dinâmica."
    else: # nni_valor > 1.2
        titulo = "📉 **Padrão Disperso (Desafio Logístico)**"
        obs = f"{prefixo} dos cortes está **muito espalhado** pela área de atuação, com poucos ou nenhum hotspot."
        acao = f"**Ação Recomendada:** Planeje as rotas com antecedência para minimizar os custos de deslocamento. Considere agrupar atendimentos por setor em dias específicos."

    return f"""
    <div style="background-color:#f0f2f6; padding: 15px; border-radius: 10px;">
    <h4 style="color:#31333f;">{titulo}</h4>
    <ul style="color:#31333f;">
        <li><b>Observação:</b> {obs}</li>
        <li><b>Ação Recomendada:</b> {acao}</li>
    </ul>
    </div>
    """

# ==============================================================================
# 4. LÓGICA PRINCIPAL DA APLICAÇÃO
# ==============================================================================
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("Escolha a planilha de cortes", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df_completo = carregar_dados(uploaded_file)
    if df_completo is not None:
        st.sidebar.success(f"{len(df_completo)} registros carregados!")
        st.sidebar.markdown("### Filtros da Análise")
        
        filtros = ['sucursal', 'centro_operativo', 'corte_recorte', 'prioridade']
        valores_selecionados = {}
        for coluna in filtros:
            if coluna in df_completo.columns:
                lista_unica = df_completo[coluna].dropna().unique().tolist()
                opcoes = sorted([str(item) for item in lista_unica])
                
                if coluna == 'prioridade':
                    valores_selecionados[coluna] = st.sidebar.multiselect(f"{coluna.replace('_', ' ').title()}", opcoes)
                else:
                    valores_selecionados[coluna] = st.sidebar.selectbox(f"{coluna.replace('_', ' ').title()}", ["Todos"] + opcoes)

        df_filtrado = df_completo.copy()
        for coluna, valor in valores_selecionados.items():
            if coluna in df_filtrado.columns:
                if coluna == 'prioridade':
                    if valor: df_filtrado = df_filtrado[df_filtrado[coluna].astype(str).isin(valor)]
                else:
                    if valor != "Todos": df_filtrado = df_filtrado[df_filtrado[coluna].astype(str) == valor]

        st.header("Resultados da Análise")
        
        if not df_filtrado.empty:
            st.sidebar.markdown("### Parâmetros de Análise")
            eps_cluster_km = st.sidebar.slider("Raio do Cluster (km)", 0.1, 5.0, 1.0, 0.1, help="Define o raio de busca para agrupar pontos no DBSCAN.")
            min_samples_cluster = st.sidebar.slider("Mínimo de Pontos por Cluster", 2, 20, 5, 1, help="Número mínimo de pontos para formar um hotspot.")
            radius_heatmap = st.sidebar.slider("Raio do Mapa de Calor (pixels)", 1, 30, 15, 1, help="Define o raio de influência de cada ponto no mapa de calor.")

            gdf_com_clusters = executar_dbscan(
                gpd.GeoDataFrame(df_filtrado, geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude), crs="EPSG:4326"),
                eps_km=eps_cluster_km, 
                min_samples=min_samples_cluster
            )
            
            # ===============================================================
            # REORDENAÇÃO DAS ABAS
            # ===============================================================
            tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Análise Geográfica e Mapa", "📊 Resumo por Centro Operativo", "🔥 Mapa de Calor", "💡 Metodologia"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total de Cortes Carregados", len(df_completo))
                col2.metric("Cortes na Seleção Atual", len(df_filtrado))
                
                nni_valor_final = None; is_media_nni = False
                centros_operativos_selecionados = gdf_com_clusters['centro_operativo'].unique()
                if len(centros_operativos_selecionados) == 1 or len(gdf_com_clusters) < 5000:
                    nni_valor_final, nni_texto = calcular_nni_otimizado(gdf_com_clusters)
                else:
                    is_media_nni = True; resultados_nni = []; pesos = []
                    for co in centros_operativos_selecionados:
                        gdf_co = gdf_com_clusters[gdf_com_clusters['centro_operativo'] == co]
                        if len(gdf_co) > 10:
                            nni, texto = calcular_nni_otimizado(gdf_co)
                            if nni is not None: resultados_nni.append(nni); pesos.append(len(gdf_co))
                    if resultados_nni:
                        nni_valor_final = np.average(resultados_nni, weights=pesos)
                        if nni_valor_final < 1: nni_texto = f"Agrupado (Média: {nni_valor_final:.2f})"
                        elif nni_valor_final > 1: nni_texto = f"Disperso (Média: {nni_valor_final:.2f})"
                        else: nni_texto = f"Aleatório (Média: {nni_valor_final:.2f})"
                    else: nni_texto = "Insuficiente para cálculo"
                
                help_nni = "O Índice do Vizinho Mais Próximo (NNI) mede se o padrão dos pontos é agrupado, disperso ou aleatório. NNI < 1: Agrupado (pontos mais próximos que o esperado). NNI ≈ 1: Aleatório (sem padrão). NNI > 1: Disperso (pontos mais espalhados que o esperado)."
                col3.metric("Padrão de Dispersão (NNI)", nni_texto, help=help_nni)
                
                n_clusters_total = len(set(gdf_com_clusters['cluster'])) - (1 if -1 in gdf_com_clusters['cluster'] else 0)
                total_pontos = len(gdf_com_clusters)
                n_ruido = list(gdf_com_clusters['cluster']).count(-1)
                percent_dispersos = (n_ruido / total_pontos * 100) if total_pontos > 0 else 0

                with st.expander("🔍 O que estes números significam? Clique para ver a análise", expanded=True):
                     resumo_html = gerar_resumo_didatico(nni_valor_final, n_clusters_total, percent_dispersos, is_media=is_media_nni)
                     st.markdown(resumo_html, unsafe_allow_html=True)

                st.subheader("Resumo da Análise de Cluster")
                n_agrupados = total_pontos - n_ruido
                if total_pontos > 0:
                    percent_agrupados = (n_agrupados / total_pontos) * 100
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Nº de Hotspots (Clusters)", f"{n_clusters_total}")
                    sub_c1, sub_c2 = st.columns(2)
                    sub_c1.metric("Nº Agrupados", f"{n_agrupados}", help="Total de serviços que fazem parte de um hotspot.")
                    sub_c1.metric("% Agrupados", f"{percent_agrupados:.1f}%")
                    sub_c2.metric("Nº Dispersos", f"{n_ruido}", help="Total de serviços isolados, que não pertencem a nenhum hotspot.")
                    sub_c2.metric("% Dispersos", f"{percent_dispersos:.1f}%")
                
                st.subheader(f"Mapa Interativo de Hotspots")
                st.write("Dê zoom no mapa para expandir os agrupamentos e ver os pontos individuais.")
                
                if not gdf_com_clusters.empty:
                    map_center = [gdf_com_clusters.latitude.mean(), gdf_com_clusters.longitude.mean()]
                    m = folium.Map(location=map_center, zoom_start=11)
                    marker_cluster = MarkerCluster().add_to(m)
                    for idx, row in gdf_com_clusters.iterrows():
                        popup_text = ""
                        for col in ['prioridade', 'centro_operativo', 'corte_recorte']:
                            if col in row: popup_text += f"{col.replace('_', ' ').title()}: {str(row[col])}<br>"
                        folium.Marker(location=[row['latitude'], row['longitude']], popup=popup_text).add_to(marker_cluster)
                    st_folium(m, use_container_width=True, height=700, returned_objects=[])

            with tab2:
                st.subheader("Análise de Cluster por Centro Operativo")
                resumo_co = gdf_com_clusters.groupby('centro_operativo').apply(lambda x: pd.Series({
                    'Total de Serviços': len(x),
                    'Nº de Clusters': x[x['cluster'] != -1]['cluster'].nunique(),
                    'Nº Agrupados': len(x[x['cluster'] != -1]),
                    'Nº Dispersos': len(x[x['cluster'] == -1])
                })).reset_index()
                resumo_co['% Agrupados'] = (resumo_co['Nº Agrupados'] / resumo_co['Total de Serviços'] * 100).round(1)
                resumo_co['% Dispersos'] = (resumo_co['Nº Dispersos'] / resumo_co['Total de Serviços'] * 100).round(1)
                resumo_co = resumo_co[['centro_operativo', 'Total de Serviços', 'Nº de Clusters', 'Nº Agrupados', '% Agrupados', 'Nº Dispersos', '% Dispersos']]
                st.dataframe(resumo_co, use_container_width=True)
            
            with tab3:
                st.subheader("Mapa de Calor dos Serviços")
                st.write("Visualize as áreas de maior concentração de serviços através de um mapa de calor. Áreas mais vermelhas indicam maior densidade de chamados.")
                if not df_filtrado.empty:
                    map_center_heatmap = [df_filtrado.latitude.mean(), df_filtrado.longitude.mean()]
                    m_heatmap = folium.Map(location=map_center_heatmap, zoom_start=11)
                    # ===============================================================
                    # CORREÇÃO PARA O MAPA DE CALOR
                    # ===============================================================
                    heat_data = df_filtrado[['latitude', 'longitude']].values.tolist()
                    HeatMap(heat_data, radius=radius_heatmap).add_to(m_heatmap)
                    st_folium(m_heatmap, use_container_width=True, height=700, returned_objects=[])
                else:
                    st.info("Nenhum dado para exibir no mapa de calor com os filtros atuais.")
            
            with tab4:
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown("""
                Para garantir uma análise precisa e confiável, utilizamos duas técnicas complementares da estatística espacial:
                
                #### 1. Clustering Baseado em Densidade (DBSCAN)
                ... [texto da metodologia] ...
                """)
                st.subheader("Perguntas Frequentes (FAQ)")
                st.markdown("""
                #### O agrupamento dos serviços é definido por "círculos"? ...
                ... [texto do FAQ] ...
                """)
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")
else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")
