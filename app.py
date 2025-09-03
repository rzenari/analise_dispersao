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
from folium.plugins import MarkerCluster
from shapely.geometry import Polygon
import requests # Nova importação para chamadas de API
from datetime import datetime

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
def carregar_dados_completos(arquivo_enviado):
    """Lê o arquivo completo com todas as colunas, que será a fonte única de dados."""
    arquivo_enviado.seek(0)
    
    def processar_dataframe(df):
        df.columns = df.columns.str.lower().str.strip()
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.error("ERRO: Colunas 'latitude' e/ou 'longitude' não foram encontradas."); st.write("Colunas encontradas:", df.columns.tolist())
            return None
        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df

    try:
        df = pd.read_csv(arquivo_enviado, encoding='utf-16', sep='\t')
        st.success("Arquivo CSV lido com sucesso (codificação: utf-16)."); return processar_dataframe(df)
    except Exception:
        try:
            arquivo_enviado.seek(0)
            df = pd.read_csv(arquivo_enviado, encoding='latin-1', sep=None, engine='python')
            st.success("Arquivo CSV lido com sucesso (codificação: latin-1)."); return processar_dataframe(df)
        except Exception:
            try:
                arquivo_enviado.seek(0)
                df = pd.read_excel(arquivo_enviado, engine='openpyxl')
                st.success("Arquivo Excel lido com sucesso."); return processar_dataframe(df)
            except Exception as e:
                st.error(f"Não foi possível ler o arquivo. Último erro: {e}"); return None

# ==============================================================================
# NOVAS FUNÇÕES PARA A PREVISÃO DO TEMPO
# ==============================================================================
@st.cache_data
def get_weather_forecast(lat, lon):
    """Busca a previsão do tempo na API da OpenWeatherMap."""
    try:
        # Tenta buscar a chave de API dos secrets do Streamlit
        api_key = st.secrets["OPENWEATHER_API_KEY"]
    except:
        # Se falhar, exibe um erro claro
        st.error("Chave de API da OpenWeatherMap não configurada. Por favor, adicione a chave 'OPENWEATHER_API_KEY' nos 'Secrets' da sua aplicação Streamlit.")
        return None

    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,alerts&appid={api_key}&units=metric&lang=pt_br"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Lança um erro para respostas ruins (4xx ou 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Não foi possível buscar a previsão do tempo: {e}")
        return None

def get_weather_icon(weather_description):
    """Retorna um ícone apropriado com base na descrição do tempo."""
    desc = weather_description.lower()
    if "chuva" in desc or "chuvisco" in desc:
        return 'tint', 'blue'
    elif "trovoada" in desc:
        return 'bolt', 'orange'
    elif "neve" in desc:
        return 'snowflake-o', 'white'
    elif "nuvens" in desc or "nublado" in desc:
        return 'cloud', 'gray'
    elif "névoa" in desc or "nevoeiro" in desc:
        return 'bars', 'lightgray'
    else: # Céu limpo, ensolarado
        return 'sun-o', 'orange'

# ==============================================================================
# FUNÇÕES DE ANÁLISE EXISTENTES (sem alterações)
# ==============================================================================
def calcular_nni_otimizado(gdf):
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
    if gdf.empty or len(gdf) < min_samples: gdf['cluster'] = -1; return gdf
    gdf_copy = gdf.copy()
    raio_terra_km = 6371; eps_rad = eps_km / raio_terra_km
    coords = np.radians(gdf_copy[['latitude', 'longitude']].values)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    gdf_copy['cluster'] = db.labels_
    return gdf_copy

def gerar_resumo_didatico(nni_valor, n_clusters, percent_dispersos, is_media=False):
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
    else:
        titulo = "📉 **Padrão Disperso (Desafio Logístico)**"
        obs = f"{prefixo} dos cortes está **muito espalhado** pela área de atuação, com poucos ou nenhum hotspot."
        acao = f"**Ação Recomendada:** Planeje as rotas com antecedência para minimizar os custos de deslocamento. Considere agrupar atendimentos por setor em dias específicos."
    return f"""<div style="background-color:#f0f2f6; padding: 15px; border-radius: 10px;"><h4 style="color:#31333f;">{titulo}</h4><ul style="color:#31333f;"><li><b>Observação:</b> {obs}</li><li><b>Ação Recomendada:</b> {acao}</li></ul></div>"""

# ==============================================================================
# 4. LÓGICA PRINCIPAL DA APLICAÇÃO
# ==============================================================================
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("Escolha a planilha de cortes", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    df_completo = carregar_dados_completos(uploaded_file)
    
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
            st.sidebar.markdown("### Parâmetros de Cluster")
            eps_cluster_km = st.sidebar.slider("Raio do Cluster (km)", 0.1, 5.0, 1.0, 0.1, help="Define o raio de busca para agrupar pontos no DBSCAN.")
            min_samples_cluster = st.sidebar.slider("Mínimo de Pontos por Cluster", 2, 20, 20, 1, help="Número mínimo de pontos para formar um hotspot.")

            gdf_base = gpd.GeoDataFrame(df_filtrado, geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude), crs="EPSG:4326")
            gdf_com_clusters = executar_dbscan(gdf_base, eps_km=eps_cluster_km, min_samples=min_samples_cluster)
            
            st.sidebar.markdown("### Filtro de Visualização do Mapa")
            tipo_visualizacao = st.sidebar.radio("Mostrar nos mapas:", ("Todos os Serviços", "Apenas Agrupados", "Apenas Dispersos"), help="Isto afeta apenas os pontos mostrados nos mapas, não as métricas.")

            gdf_visualizacao = gdf_com_clusters.copy()
            if tipo_visualizacao == "Apenas Agrupados":
                gdf_visualizacao = gdf_com_clusters[gdf_com_clusters['cluster'] != -1]
            elif tipo_visualizacao == "Apenas Dispersos":
                gdf_visualizacao = gdf_com_clusters[gdf_com_clusters['cluster'] == -1]
            
            if 'numero_ordem' in gdf_com_clusters.columns:
                st.sidebar.markdown("### 📥 Downloads")
                df_agrupados_download = gdf_com_clusters[gdf_com_clusters['cluster'] != -1].drop(columns=['geometry'])
                df_dispersos_download = gdf_com_clusters[gdf_com_clusters['cluster'] == -1].drop(columns=['geometry'])
                csv_agrupados = df_agrupados_download.to_csv(index=False).encode('utf-8-sig')
                csv_dispersos = df_dispersos_download.to_csv(index=False).encode('utf-8-sig')
                st.sidebar.download_button(label="⬇️ Baixar Serviços Agrupados", data=csv_agrupados, file_name='servicos_agrupados.csv', mime='text/csv', disabled=df_agrupados_download.empty)
                st.sidebar.download_button(label="⬇️ Baixar Serviços Dispersos", data=csv_dispersos, file_name='servicos_dispersos.csv', mime='text/csv', disabled=df_dispersos_download.empty)

            tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Análise Geográfica e Mapa", "📊 Resumo por Centro Operativo", "📍 Contorno dos Clusters", "💡 Metodologia"])

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
                help_nni = "O Índice do Vizinho Mais Próximo (NNI) mede se o padrão dos pontos é agrupado, disperso ou aleatório. NNI < 1: Agrupado. NNI ≈ 1: Aleatório. NNI > 1: Disperso."
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
                if not gdf_visualizacao.empty:
                    map_center = [gdf_visualizacao.latitude.mean(), gdf_visualizacao.longitude.mean()]
                    m = folium.Map(location=map_center, zoom_start=11)
                    
                    # ADICIONANDO MARCADOR DE PREVISÃO DO TEMPO
                    weather_data = get_weather_forecast(map_center[0], map_center[1])
                    if weather_data:
                        current_weather = weather_data.get('current', {})
                        daily_forecast = weather_data.get('daily', [{}])[0]
                        
                        temp_atual = current_weather.get('temp', 'N/A')
                        desc_atual = current_weather.get('weather', [{}])[0].get('description', 'N/A').capitalize()
                        
                        temp_amanha = daily_forecast.get('temp', {}).get('day', 'N/A')
                        desc_amanha = daily_forecast.get('summary', 'Previsão não disponível.')
                        chuva_amanha = daily_forecast.get('pop', 0) * 100

                        icon_name, icon_color = get_weather_icon(desc_atual)

                        popup_html = f"""
                        <b>📍 Previsão para a Região</b><br>
                        <hr style='margin: 5px 0;'>
                        <b>Agora:</b> {temp_atual}°C, {desc_atual}<br>
                        <b>Amanhã:</b> {temp_amanha}°C, {desc_amanha}<br>
                        <b>Chance de Chuva (Amanhã):</b> {chuva_amanha:.0f}%
                        """
                        
                        folium.Marker(
                            location=map_center,
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip="Clique para ver a previsão do tempo",
                            icon=folium.Icon(color=icon_color, icon=icon_name, prefix='fa')
                        ).add_to(m)

                    marker_cluster = MarkerCluster().add_to(m)
                    for idx, row in gdf_visualizacao.iterrows():
                        popup_text = ""
                        for col in ['prioridade', 'centro_operativo', 'corte_recorte']:
                            if col in row: popup_text += f"{col.replace('_', ' ').title()}: {str(row[col])}<br>"
                        folium.Marker(location=[row['latitude'], row['longitude']], popup=popup_text).add_to(marker_cluster)
                    st_folium(m, use_container_width=True, height=700, returned_objects=[])
                else:
                    st.warning("Nenhum serviço para exibir no mapa com o filtro de visualização atual.")

            with tab2:
                #... (conteúdo da tab2, sem alterações)
                st.subheader("Análise de Cluster por Centro Operativo")
                resumo_co = gdf_com_clusters.groupby('centro_operativo').apply(lambda x: pd.Series({
                    'Total de Serviços': len(x),
                    'Nº de Clusters': x[x['cluster'] != -1]['cluster'].nunique(),
                    'Nº Agrupados': len(x[x['cluster'] != -1]),
                    'Nº Dispersos': len(x[x['cluster'] == -1])
                }), include_groups=False).reset_index()
                resumo_co['% Agrupados'] = (resumo_co['Nº Agrupados'] / resumo_co['Total de Serviços'] * 100).round(1)
                resumo_co['% Dispersos'] = (resumo_co['Nº Dispersos'] / resumo_co['Total de Serviços'] * 100).round(1)
                resumo_co = resumo_co[['centro_operativo', 'Total de Serviços', 'Nº de Clusters', 'Nº Agrupados', '% Agrupados', 'Nº Dispersos', '% Dispersos']]
                st.dataframe(resumo_co, use_container_width=True)
            
            with tab3:
                #... (conteúdo da tab3, sem alterações)
                st.subheader("Contorno Geográfico dos Clusters")
                st.write("Este mapa desenha um polígono (contorno) ao redor de cada hotspot identificado, mostrando a área geográfica exata de cada agrupamento.")
                gdf_clusters_reais = gdf_visualizacao[gdf_visualizacao['cluster'] != -1]
                
                if not gdf_clusters_reais.empty:
                    map_center_hull = [gdf_clusters_reais.latitude.mean(), gdf_clusters_reais.longitude.mean()]
                    m_hull = folium.Map(location=map_center_hull, zoom_start=11)
                    
                    try:
                        counts = gdf_clusters_reais.groupby('cluster').size().rename('contagem')
                        hulls = gdf_clusters_reais.dissolve(by='cluster').convex_hull
                        gdf_hulls = gpd.GeoDataFrame(geometry=hulls).reset_index()
                        gdf_hulls_proj = gdf_hulls.to_crs("EPSG:3857")
                        gdf_hulls['area_km2'] = (gdf_hulls_proj.geometry.area / 1_000_000).round(2)
                        gdf_hulls = gdf_hulls.merge(counts, on='cluster')
                        gdf_hulls['densidade'] = (gdf_hulls['contagem'] / gdf_hulls['area_km2']).round(1)
                        
                        folium.GeoJson(
                            gdf_hulls,
                            style_function=lambda x: {'color': 'red', 'weight': 2, 'fillColor': 'red', 'fillOpacity': 0.2},
                            tooltip=folium.GeoJsonTooltip(
                                fields=['cluster', 'contagem', 'area_km2', 'densidade'],
                                aliases=['Cluster ID:', 'Nº de Serviços:', 'Área (km²):', 'Serviços por km²:'],
                                localize=True,
                                sticky=True
                            )
                        ).add_to(m_hull)
                        
                        marker_cluster_hull = MarkerCluster().add_to(m_hull)
                        for idx, row in gdf_clusters_reais.iterrows():
                            popup_text = f"Cluster: {row['cluster']}"
                            folium.Marker(location=[row['latitude'], row['longitude']], popup=popup_text, icon=folium.Icon(color='blue', icon='info-sign')).add_to(marker_cluster_hull)

                        st_folium(m_hull, use_container_width=True, height=700, returned_objects=[])
                    except Exception as e:
                        st.warning(f"Não foi possível desenhar os contornos dos clusters. Isso pode acontecer se um cluster tiver poucos pontos para formar uma área. Erro: {e}")
                else:
                    st.warning("Nenhum cluster para desenhar com os filtros atuais.")

            with tab4:
                #... (conteúdo da tab4, sem alterações)
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown("""
                ... [texto completo da metodologia] ...
                """)
                st.subheader("Perguntas Frequentes (FAQ)")
                st.markdown("""
                ... [texto completo do FAQ] ...
                """)
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")
else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")
