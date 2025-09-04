# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import cKDTree
from math import sqrt
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from shapely.geometry import Polygon

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E TÍTULOS
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análise de Dispersão Geográfica")

st.title("🗺️ Ferramenta de Análise de Dispersão Geográfica")
st.write("Faça o upload da sua planilha de cortes para analisar a distribuição geográfica e identificar clusters")

# ==============================================================================
# 3. FUNÇÕES DE ANÁLISE (CACHE REMOVIDO PARA GARANTIR REATIVIDADE)
# ==============================================================================

# @st.cache_data # Cache removido
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
        st.success("Arquivo CSV de cortes lido com sucesso (codificação: utf-16)."); return processar_dataframe(df)
    except Exception:
        try:
            arquivo_enviado.seek(0)
            df = pd.read_csv(arquivo_enviado, encoding='latin-1', sep=None, engine='python')
            st.success("Arquivo CSV de cortes lido com sucesso (codificação: latin-1)."); return processar_dataframe(df)
        except Exception:
            try:
                arquivo_enviado.seek(0)
                df = pd.read_excel(arquivo_enviado, engine='openpyxl')
                st.success("Arquivo Excel de cortes lido com sucesso."); return processar_dataframe(df)
            except Exception as e:
                st.error(f"Não foi possível ler o arquivo de cortes. Último erro: {e}"); return None

# @st.cache_data # Cache removido
def carregar_dados_metas(arquivo_metas):
    """Lê o arquivo opcional de metas e equipes."""
    if arquivo_metas is None:
        return None
    arquivo_metas.seek(0)
    try:
        df = pd.read_excel(arquivo_metas, engine='openpyxl')
        df.columns = df.columns.str.lower().str.strip()
        if 'centro_operativo' in df.columns:
            return df
        else:
            st.error("ERRO: A planilha de metas precisa conter uma coluna chamada 'Centro_Operativo'.")
            return None
    except Exception as e:
        st.error(f"Não foi possível ler a planilha de metas. Erro: {e}")
        return None


def calcular_nni_otimizado(gdf):
    """Calcula NNI de forma otimizada para memória."""
    if len(gdf) < 3: return None, "Pontos insuficientes (< 3)."
    n_points = len(gdf)
    points = np.array([gdf.geometry.x, gdf.geometry.y]).T
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    observed_mean_dist = np.mean(distances[:, 1])
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
    if gdf.empty or len(gdf) < min_samples: 
        gdf['cluster'] = -1
        return gdf
    
    gdf_copy = gdf.copy()
    raio_terra_km = 6371; eps_rad = eps_km / raio_terra_km
    coords = np.radians(gdf_copy[['latitude', 'longitude']].values)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords)
    gdf_copy['cluster'] = db.labels_
    return gdf_copy

def simular_pacotes_de_trabalho(gdf_cluster, n_equipes, capacidade):
    """Simula a criação de pacotes de trabalho usando K-Means e realocação para respeitar a capacidade."""
    if gdf_cluster.empty or n_equipes == 0:
        return gdf_cluster.copy(), gpd.GeoDataFrame()

    kmeans = KMeans(n_clusters=n_equipes, random_state=42, n_init='auto')
    coords = gdf_cluster[['longitude', 'latitude']].values
    gdf_cluster['pacote_id'] = kmeans.fit_predict(coords)
    
    pacotes_sobrecarregados = []
    indices_excedentes = []

    for i in range(n_equipes):
        pacote_atual = gdf_cluster[gdf_cluster['pacote_id'] == i]
        if len(pacote_atual) > capacidade:
            pacotes_sobrecarregados.append(i)
            centro = kmeans.cluster_centers_[i]
            distancias = np.linalg.norm(pacote_atual[['longitude', 'latitude']].values - centro, axis=1)
            pacote_atual_sorted = pacote_atual.iloc[distancias.argsort()]
            indices_excedentes.extend(pacote_atual_sorted.index[capacidade:])

    if indices_excedentes:
        gdf_excedentes = gdf_cluster.loc[indices_excedentes].copy()
        gdf_cluster.loc[indices_excedentes, 'pacote_id'] = -1
        
        if not gdf_excedentes.empty:
            tree_excedentes = cKDTree(gdf_excedentes[['longitude', 'latitude']].values)

            for i in range(n_equipes):
                if i not in pacotes_sobrecarregados:
                    pacote_atual = gdf_cluster[gdf_cluster['pacote_id'] == i]
                    capacidade_restante = capacidade - len(pacote_atual)
                    
                    if capacidade_restante > 0 and not gdf_excedentes.empty:
                        centro_pacote = kmeans.cluster_centers_[i]
                        dist, idx = tree_excedentes.query([centro_pacote], k=min(capacidade_restante, len(gdf_excedentes)))
                        
                        # idx pode ser um único valor ou um array, normalizamos para array
                        if not isinstance(idx, np.ndarray): idx = np.array([idx])
                        
                        indices_para_alocar = gdf_excedentes.index[idx.flatten()]
                        
                        gdf_cluster.loc[indices_para_alocar, 'pacote_id'] = i
                        gdf_excedentes = gdf_excedentes.drop(indices_para_alocar)

                        if not gdf_excedentes.empty:
                            tree_excedentes = cKDTree(gdf_excedentes[['longitude', 'latitude']].values)
    
    gdf_alocados = gdf_cluster[gdf_cluster['pacote_id'] != -1].copy()
    gdf_excedentes_final = gdf_cluster[gdf_cluster['pacote_id'] == -1].copy()
    
    return gdf_alocados, gdf_excedentes_final


def gerar_resumo_didatico(nni_valor, n_clusters, percent_dispersos, is_media=False):
    """Gera um texto interpretativo com base nos resultados da análise."""
    if nni_valor is None: return ""
    prefixo = "Na média, o padrão" if is_media else "O padrão"
    if percent_dispersos > 50:
        titulo = "⚠️ **Padrão Misto (Agrupamentos Isolados)**"; obs = f"Apesar da existência de **{n_clusters} hotspots**, a maioria dos serviços (**{percent_dispersos:.1f}%**) está **dispersa** pela região."; acao = f"**Ação Recomendada:** Trate a operação de forma híbrida. Otimize rotas para os hotspots e agrupe os serviços dispersos por setor ou dia."
    elif nni_valor < 0.5:
        titulo = "📈 **Padrão Fortemente Agrupado (Excelente Oportunidade Logística)**"; obs = f"{prefixo} dos cortes é **fortemente concentrado** em áreas específicas."; acao = f"**Ação Recomendada:** Crie rotas otimizadas com baixo deslocamento. Avalie alocar equipes dedicadas para os **{n_clusters} hotspots** encontrados."
    elif 0.5 <= nni_valor < 0.8:
        titulo = "📊 **Padrão Moderadamente Agrupado (Potencial de Otimização)**"; obs = f"{prefixo} dos cortes apresenta **boa concentração**."; acao = f"**Ação Recomendada:** Identifique os **{n_clusters} hotspots** mais densos para priorizar o roteamento."
    elif 0.8 <= nni_valor <= 1.2:
        titulo = "😐 **Padrão Aleatório (Sem Padrão Claro)**"; obs = f"{prefixo} dos cortes é **aleatório**."; acao = f"**Ação Recomendada:** A logística tende a ser menos previsível. Considere uma abordagem de roteirização diária e dinâmica."
    else: 
        titulo = "📉 **Padrão Disperso (Desafio Logístico)**"; obs = f"{prefixo} dos cortes está **muito espalhado**."; acao = f"**Ação Recomendada:** Planeje rotas com antecedência para minimizar custos de deslocamento."
    return f"""<div style="background-color:#f0f2f6; padding: 15px; border-radius: 10px;"><h4 style="color:#31333f;">{titulo}</h4><ul style="color:#31333f;"><li><b>Observação:</b> {obs}</li><li><b>Ação Recomendada:</b> {acao}</li></ul></div>"""

def calcular_qualidade_carteira(row):
    """Calcula a qualidade da carteira com base nas metas e serviços agrupados."""
    if pd.isna(row.get('meta diária')) or row.get('meta diária') == 0: return "Sem Meta"
    if pd.isna(row.get('nº agrupados')) or pd.isna(row.get('total de serviços')): return "Dados Insuficientes"
    if row['nº agrupados'] >= row['meta diária']: return "✅ Ótima"
    elif row['total de serviços'] >= row['meta diária']: return "⚠️ Atenção"
    else: return "❌ Crítica"

# ==============================================================================
# 4. LÓGICA PRINCIPAL DA APLICAÇÃO
# ==============================================================================
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("1. Escolha a planilha de cortes", type=["csv", "xlsx", "xls"])
metas_file = st.sidebar.file_uploader("2. Escolha a planilha de metas (Opcional)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df_completo = carregar_dados_completos(uploaded_file)
    df_metas = carregar_dados_metas(metas_file)
    
    if df_completo is not None:
        if not st.sidebar.button('Recarregar Dados'):
            st.sidebar.success(f"{len(df_completo)} registros carregados!")
            if df_metas is not None: st.sidebar.info(f"Metas carregadas para {len(df_metas)} COs.")

        st.sidebar.markdown("### Filtros da Análise")
        filtros = ['sucursal', 'centro_operativo', 'corte_recorte', 'prioridade']
        valores_selecionados = {}
        for coluna in filtros:
            if coluna in df_completo.columns:
                lista_unica = df_completo[coluna].dropna().unique().tolist()
                opcoes = sorted([str(item) for item in lista_unica])
                if coluna == 'prioridade': valores_selecionados[coluna] = st.sidebar.multiselect(f"{coluna.replace('_', ' ').title()}", opcoes)
                else: valores_selecionados[coluna] = st.sidebar.selectbox(f"{coluna.replace('_', ' ').title()}", ["Todos"] + opcoes)

        df_filtrado = df_completo.copy()
        for coluna, valor in valores_selecionados.items():
            if coluna in df_filtrado.columns:
                if coluna == 'prioridade':
                    if valor: df_filtrado = df_filtrado[df_filtrado[coluna].astype(str).isin(valor)]
                elif valor != "Todos": df_filtrado = df_filtrado[df_filtrado[coluna].astype(str) == valor]

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
            if tipo_visualizacao == "Apenas Agrupados": gdf_visualizacao = gdf_com_clusters[gdf_com_clusters['cluster'] != -1]
            elif tipo_visualizacao == "Apenas Dispersos": gdf_visualizacao = gdf_com_clusters[gdf_com_clusters['cluster'] == -1]
            
            if 'numero_ordem' in gdf_com_clusters.columns:
                st.sidebar.markdown("### 📥 Downloads")
                df_agrupados_download = gdf_com_clusters[gdf_com_clusters['cluster'] != -1].drop(columns=['geometry']); df_dispersos_download = gdf_com_clusters[gdf_com_clusters['cluster'] == -1].drop(columns=['geometry'])
                csv_agrupados = df_agrupados_download.to_csv(index=False).encode('utf-8-sig'); csv_dispersos = df_dispersos_download.to_csv(index=False).encode('utf-8-sig')
                st.sidebar.download_button(label="⬇️ Baixar Serviços Agrupados", data=csv_agrupados, file_name='servicos_agrupados.csv', mime='text/csv', disabled=df_agrupados_download.empty)
                st.sidebar.download_button(label="⬇️ Baixar Serviços Dispersos", data=csv_dispersos, file_name='servicos_dispersos.csv', mime='text/csv', disabled=df_dispersos_download.empty)

            lista_abas = ["🗺️ Análise Geográfica", "📊 Resumo por CO", "📍 Contorno dos Clusters"]
            if df_metas is not None: lista_abas.append("📦 Pacotes de Trabalho")
            lista_abas.append("💡 Metodologia")
            tabs = st.tabs(lista_abas)

            with tabs[0]: # Análise Geográfica
                col1, col2, col3 = st.columns(3); col1.metric("Total de Cortes Carregados", len(df_completo)); col2.metric("Cortes na Seleção Atual", len(df_filtrado))
                nni_valor_final, nni_texto = calcular_nni_otimizado(gdf_com_clusters)
                help_nni = "O Índice do Vizinho Mais Próximo (NNI) mede se o padrão dos pontos é agrupado, disperso ou aleatório. NNI < 1: Agrupado. NNI ≈ 1: Aleatório. NNI > 1: Disperso."
                col3.metric("Padrão de Dispersão (NNI)", nni_texto, help=help_nni)
                n_clusters_total = len(set(gdf_com_clusters['cluster'])) - (1 if -1 in gdf_com_clusters['cluster'] else 0)
                total_pontos = len(gdf_com_clusters); n_ruido = list(gdf_com_clusters['cluster']).count(-1); percent_dispersos = (n_ruido / total_pontos * 100) if total_pontos > 0 else 0
                with st.expander("🔍 O que estes números significam?", expanded=True):
                     st.markdown(gerar_resumo_didatico(nni_valor_final, n_clusters_total, percent_dispersos), unsafe_allow_html=True)
                st.subheader("Resumo da Análise de Cluster")
                n_agrupados = total_pontos - n_ruido
                if total_pontos > 0:
                    percent_agrupados = (n_agrupados / total_pontos) * 100
                    c1,c2,c3 = st.columns(3);c1.metric("Nº de Hotspots", f"{n_clusters_total}")
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Nº Agrupados", f"{n_agrupados}"); sc1.metric("% Agrupados", f"{percent_agrupados:.1f}%")
                    sc2.metric("Nº Dispersos", f"{n_ruido}"); sc2.metric("% Dispersos", f"{percent_dispersos:.1f}%")
                st.subheader(f"Mapa Interativo de Hotspots"); st.write("Dê zoom no mapa para expandir os agrupamentos.")
                if not gdf_visualizacao.empty:
                    map_center = [gdf_visualizacao.latitude.mean(), gdf_visualizacao.longitude.mean()]; m = folium.Map(location=map_center, zoom_start=11)
                    marker_cluster = MarkerCluster().add_to(m)
                    for idx, row in gdf_visualizacao.iterrows():
                        popup_text = "".join([f"{col.replace('_', ' ').title()}: {str(row[col])}<br>" for col in ['prioridade', 'centro_operativo', 'corte_recorte'] if col in row])
                        folium.Marker(location=[row['latitude'], row['longitude']], popup=popup_text).add_to(marker_cluster)
                    st_folium(m, use_container_width=True, height=700)
                else: st.warning("Nenhum serviço para exibir no mapa.")

            with tabs[1]: # Resumo por CO
                st.subheader("Análise de Cluster por Centro Operativo")
                resumo_co = gdf_com_clusters.groupby('centro_operativo').apply(lambda x: pd.Series({'total de serviços': len(x), 'nº de clusters': x[x['cluster'] != -1]['cluster'].nunique(), 'nº agrupados': len(x[x['cluster'] != -1]),'nº dispersos': len(x[x['cluster'] == -1])}), include_groups=False).reset_index()
                resumo_co['% agrupados'] = (resumo_co['nº agrupados'] / resumo_co['total de serviços'] * 100).round(1)
                resumo_co['% dispersos'] = (resumo_co['nº dispersos'] / resumo_co['total de serviços'] * 100).round(1)
                if df_metas is not None:
                    df_metas_renamed = df_metas.rename(columns=lambda x: x.replace(' ', '_'))
                    resumo_co = pd.merge(resumo_co, df_metas_renamed, on='centro_operativo', how='left')
                    resumo_co['qualidade da carteira'] = resumo_co.apply(calcular_qualidade_carteira, axis=1)
                st.dataframe(resumo_co, use_container_width=True)

            with tabs[2]: # Contorno dos Clusters
                st.subheader("Contorno Geográfico dos Clusters"); st.write("Este mapa desenha um polígono ao redor de cada hotspot.")
                gdf_clusters_reais = gdf_visualizacao[gdf_visualizacao['cluster'] != -1]
                if not gdf_clusters_reais.empty:
                    map_center_hull = [gdf_clusters_reais.latitude.mean(), gdf_clusters_reais.longitude.mean()]; m_hull = folium.Map(location=map_center_hull, zoom_start=11)
                    try:
                        counts = gdf_clusters_reais.groupby('cluster').size().rename('contagem')
                        hulls = gdf_clusters_reais.dissolve(by='cluster').convex_hull
                        gdf_hulls = gpd.GeoDataFrame(geometry=hulls).reset_index(); gdf_hulls_proj = gdf_hulls.to_crs("EPSG:3857")
                        gdf_hulls['area_km2'] = (gdf_hulls_proj.geometry.area / 1_000_000).round(2)
                        gdf_hulls = gdf_hulls.merge(counts, on='cluster')
                        gdf_hulls['densidade'] = (gdf_hulls['contagem'] / gdf_hulls['area_km2']).round(1)
                        folium.GeoJson(gdf_hulls, style_function=lambda x: {'color': 'red', 'weight': 2, 'fillColor': 'red', 'fillOpacity': 0.2}, tooltip=folium.GeoJsonTooltip(fields=['cluster', 'contagem', 'area_km2', 'densidade'], aliases=['Cluster ID:', 'Nº de Serviços:', 'Área (km²):', 'Serviços por km²:'], localize=True, sticky=True)).add_to(m_hull)
                        marker_cluster_hull = MarkerCluster().add_to(m_hull)
                        for idx, row in gdf_clusters_reais.iterrows():
                            folium.Marker(location=[row['latitude'], row['longitude']], popup=f"Cluster: {row['cluster']}", icon=folium.Icon(color='blue', icon='info-sign')).add_to(marker_cluster_hull)
                        st_folium(m_hull, use_container_width=True, height=700)
                    except Exception as e: st.warning(f"Não foi possível desenhar os contornos. Erro: {e}")
                else: st.warning("Nenhum cluster para desenhar.")
            
            if df_metas is not None:
                with tabs[3]: # Pacotes de Trabalho
                    st.subheader("Simulação de Roteirização Diária"); st.write("Este mapa simula a alocação dos serviços agrupados entre as equipes de um CO, respeitando a capacidade de produção de cada uma.")
                    gdf_clusters_reais = gdf_com_clusters[gdf_com_clusters['cluster'] != -1]
                    if not gdf_clusters_reais.empty:
                        todos_alocados = []; todos_excedentes = []
                        for co in gdf_clusters_reais['centro_operativo'].unique():
                            gdf_co = gdf_clusters_reais[gdf_clusters_reais['centro_operativo'] == co].copy()
                            metas_co = df_metas[df_metas['centro_operativo'] == co]
                            if not metas_co.empty:
                                n_equipes = int(metas_co['equipes'].iloc[0]); capacidade = int(metas_co['produção'].iloc[0])
                                if n_equipes > 0 and capacidade > 0 and len(gdf_co) > 0:
                                    alocados, excedentes = simular_pacotes_de_trabalho(gdf_co, n_equipes, capacidade)
                                    todos_alocados.append(alocados); todos_excedentes.append(excedentes)
                        if todos_alocados:
                            gdf_alocados_final = pd.concat(todos_alocados) if todos_alocados else gpd.GeoDataFrame()
                            gdf_excedentes_final = pd.concat(todos_excedentes) if todos_excedentes else gpd.GeoDataFrame()
                            st.markdown("##### Performance da Carteira Agrupada")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Serviços Agrupados", len(gdf_clusters_reais))
                            c2.metric("Serviços Alocados", len(gdf_alocados_final))
                            c3.metric("Serviços Excedentes", len(gdf_excedentes_final), delta=f"-{len(gdf_excedentes_final)} não roteirizados", delta_color="inverse")
                            map_center_pacotes = [gdf_clusters_reais.latitude.mean(), gdf_clusters_reais.longitude.mean()]
                            m_pacotes = folium.Map(location=map_center_pacotes, zoom_start=10)
                            cores_co = {co: color for co, color in zip(gdf_com_clusters['centro_operativo'].unique(), ['blue', 'green', 'purple', 'orange', 'darkred', 'red', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'lightgreen', 'pink', 'lightblue', 'lightgray', 'black'])}
                            if not gdf_alocados_final.empty:
                                hulls_pacotes = gdf_alocados_final.dissolve(by=['centro_operativo', 'pacote_id']).convex_hull
                                gdf_hulls_pacotes = gpd.GeoDataFrame(geometry=hulls_pacotes).reset_index()
                                folium.GeoJson(gdf_hulls_pacotes, style_function=lambda feature: {'color': cores_co.get(feature['properties']['centro_operativo'], 'gray'), 'weight': 2.5, 'fillColor': cores_co.get(feature['properties']['centro_operativo'], 'gray'), 'fillOpacity': 0.25}, tooltip=f"CO: {feature['properties']['centro_operativo']}, Pacote: {feature['properties']['pacote_id']}").add_to(m_pacotes)
                            for _, row in gdf_excedentes_final.iterrows():
                                folium.Marker(location=[row['latitude'], row['longitude']], tooltip="Serviço Excedente", icon=folium.Icon(color='red', icon='times-circle', prefix='fa')).add_to(m_pacotes)
                            st_folium(m_pacotes, use_container_width=True, height=700)
                        else: st.info("Nenhum pacote de trabalho para simular.")
                    else: st.warning("Nenhum cluster encontrado para dividir em pacotes.")
            
            with tabs[-1]: # Metodologia
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown("""[Texto completo da metodologia]""")
                st.subheader("Perguntas Frequentes (FAQ)")
                st.markdown("""[Texto completo do FAQ]""")
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")
else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")
