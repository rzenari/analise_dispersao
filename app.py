# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import cKDTree
from math import sqrt, ceil
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
# 3. FUNÇÕES DE ANÁLISE
# ==============================================================================
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

def carregar_dados_metas(arquivo_metas):
    """Lê o arquivo opcional de metas e equipes."""
    if arquivo_metas is None:
        return None
    arquivo_metas.seek(0)
    try:
        df = pd.read_excel(arquivo_metas, engine='openpyxl')
        df.columns = df.columns.str.lower().str.strip()
        if 'centro_operativo' in df.columns and 'produção' in df.columns:
            df['centro_operativo'] = df['centro_operativo'].str.strip().str.upper()
            return df
        else:
            st.error("ERRO: A planilha de metas precisa conter as colunas 'Centro_Operativo' e 'Produção'.")
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

def simular_pacotes_por_densidade(gdf_co, n_equipes, capacidade_producao):
    """
    Simula pacotes com base na densidade, respeitando estritamente a capacidade de produção.
    A subdivisão de grandes clusters agora usa um método iterativo de 'descascamento'
    para garantir que nenhum pacote exceda o limite.
    """
    if gdf_co.empty or n_equipes == 0 or capacidade_producao == 0:
        return gpd.GeoDataFrame(), gdf_co.copy()

    gdf_clusters_reais = gdf_co[gdf_co['cluster'] != -1].copy()
    if gdf_clusters_reais.empty:
        return gpd.GeoDataFrame(), gdf_co.copy()

    pacotes_candidatos = []
    
    for cluster_id in gdf_clusters_reais['cluster'].unique():
        gdf_cluster_atual = gdf_clusters_reais[gdf_clusters_reais['cluster'] == cluster_id]
        contagem = len(gdf_cluster_atual)

        if 0 < contagem <= capacidade_producao:
            pacotes_candidatos.append({'indices': gdf_cluster_atual.index, 'pontos': gdf_cluster_atual})
        
        elif contagem > capacidade_producao:
            gdf_temp = gdf_cluster_atual.copy()
            while len(gdf_temp) > 0:
                if len(gdf_temp) <= capacidade_producao:
                    pacotes_candidatos.append({'indices': gdf_temp.index, 'pontos': gdf_temp})
                    break
                
                coords_temp = gdf_temp[['longitude', 'latitude']].values
                tree = cKDTree(coords_temp)
                
                _, indices_vizinhos = tree.query(coords_temp[0], k=capacidade_producao)
                
                indices_reais_no_gdf_temp = gdf_temp.index[indices_vizinhos]
                sub_pacote = gdf_temp.loc[indices_reais_no_gdf_temp]
                
                pacotes_candidatos.append({'indices': sub_pacote.index, 'pontos': sub_pacote})
                
                gdf_temp.drop(indices_reais_no_gdf_temp, inplace=True)

    pacotes_ranqueados = []
    for candidato in pacotes_candidatos:
        pontos = candidato['pontos']
        contagem = len(pontos)
        if contagem > 0:
            try:
                hull = pontos.unary_union.convex_hull
                gdf_hull = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
                area_km2 = (gdf_hull.to_crs("EPSG:3857").geometry.area / 1_000_000).iloc[0]
                densidade = contagem / area_km2 if area_km2 > 0 else 0
                
                candidato['contagem'] = contagem
                candidato['area_km2'] = round(area_km2, 2)
                candidato['densidade'] = round(densidade, 2)
                pacotes_ranqueados.append(candidato)
            except Exception:
                continue

    pacotes_ranqueados.sort(key=lambda p: p['densidade'], reverse=True)
    pacotes_vencedores = pacotes_ranqueados[:n_equipes]
    
    indices_alocados = []
    gdf_co['pacote_id'] = -1 
    for i, pacote in enumerate(pacotes_vencedores):
        gdf_co.loc[pacote['indices'], 'pacote_id'] = i
        indices_alocados.extend(pacote['indices'])

    gdf_alocados = gdf_co.loc[indices_alocados].copy()
    gdf_excedentes = gdf_co.drop(indices_alocados).copy()

    return gdf_alocados, gdf_excedentes

def gerar_resumo_didatico(nni_valor, n_clusters, percent_dispersos, is_media=False):
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
    meta_diaria = row.get('meta_diária', 0)
    if pd.isna(meta_diaria) or meta_diaria == 0: return "Sem Meta"
    if pd.isna(row.get('nº agrupados')) or pd.isna(row.get('total de serviços')): return "Dados Insuficientes"
    if row['nº agrupados'] >= meta_diaria: return "✅ Ótima"
    elif row['total de serviços'] >= meta_diaria: return "⚠️ Atenção"
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
        st.sidebar.success(f"{len(df_completo)} registros carregados!")
        if df_metas is not None: st.sidebar.info(f"Metas carregadas para {len(df_metas)} COs.")

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

            with tabs[0]:
                with st.spinner('Carregando análise e mapa...'):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total de Cortes Carregados", len(df_completo))
                    col2.metric("Cortes na Seleção Atual", len(df_filtrado))
                    nni_valor_final, nni_texto = calcular_nni_otimizado(gdf_com_clusters)
                    help_nni = "O Índice do Vizinho Mais Próximo (NNI) mede se o padrão dos pontos é agrupado, disperso ou aleatório. NNI < 1: Agrupado. NNI ≈ 1: Aleatório. NNI > 1: Disperso."
                    col3.metric("Padrão de Dispersão (NNI)", nni_texto, help=help_nni)
                    n_clusters_total = len(set(gdf_com_clusters['cluster'])) - (1 if -1 in gdf_com_clusters['cluster'] else 0)
                    total_pontos = len(gdf_com_clusters)
                    n_ruido = list(gdf_com_clusters['cluster']).count(-1)
                    percent_dispersos = (n_ruido / total_pontos * 100) if total_pontos > 0 else 0
                    with st.expander("🔍 O que estes números significam?", expanded=True):
                        st.markdown(gerar_resumo_didatico(nni_valor_final, n_clusters_total, percent_dispersos), unsafe_allow_html=True)
                    st.subheader("Resumo da Análise de Cluster")
                    n_agrupados = total_pontos - n_ruido
                    if total_pontos > 0:
                        percent_agrupados = (n_agrupados / total_pontos) * 100
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Nº de Hotspots", f"{n_clusters_total}")
                        sc1, sc2 = st.columns(2)
                        sc1.metric("Nº Agrupados", f"{n_agrupados}")
                        sc1.metric("% Agrupados", f"{percent_agrupados:.1f}%")
                        sc2.metric("Nº Dispersos", f"{n_ruido}")
                        sc2.metric("% Dispersos", f"{percent_dispersos:.1f}%")
                    st.subheader(f"Mapa Interativo de Hotspots")
                    st.write("Dê zoom no mapa para expandir os agrupamentos.")
                    if not gdf_visualizacao.empty:
                        map_center = [gdf_visualizacao.latitude.mean(), gdf_visualizacao.longitude.mean()]
                        m = folium.Map(location=map_center, zoom_start=11)
                        marker_cluster = MarkerCluster().add_to(m)
                        for idx, row in gdf_visualizacao.iterrows():
                            popup_text = "".join([f"{col.replace('_', ' ').title()}: {str(row[col])}<br>" for col in ['prioridade', 'centro_operativo', 'corte_recorte'] if col in row])
                            folium.Marker(location=[row['latitude'], row['longitude']], popup=popup_text).add_to(marker_cluster)
                        st_folium(m, use_container_width=True, height=700)
                    else:
                        st.warning("Nenhum serviço para exibir no mapa.")

            with tabs[1]:
                with st.spinner('Gerando tabela de resumo...'):
                    st.subheader("Análise de Cluster por Centro Operativo")
                    resumo_co = gdf_com_clusters.groupby('centro_operativo').apply(lambda x: pd.Series({'total de serviços': len(x), 'nº de clusters': x[x['cluster'] != -1]['cluster'].nunique(), 'nº agrupados': len(x[x['cluster'] != -1]),'nº dispersos': len(x[x['cluster'] == -1])}), include_groups=False).reset_index()
                    resumo_co['% agrupados'] = (resumo_co['nº agrupados'] / resumo_co['total de serviços'] * 100).round(1)
                    resumo_co['% dispersos'] = (resumo_co['nº dispersos'] / resumo_co['total de serviços'] * 100).round(1)
                    if df_metas is not None:
                        resumo_co['centro_operativo_join_key'] = resumo_co['centro_operativo'].str.strip().str.upper()
                        df_metas['centro_operativo_join_key'] = df_metas['centro_operativo'].str.strip().str.upper()
                        
                        df_metas_renamed = df_metas.rename(columns=lambda x: x.replace(' ', '_'))
                        resumo_co = pd.merge(resumo_co, df_metas_renamed, on='centro_operativo_join_key', how='left').drop(columns=['centro_operativo_y', 'centro_operativo_join_key']).rename(columns={'centro_operativo_x': 'centro_operativo'})
                        resumo_co['qualidade da carteira'] = resumo_co.apply(calcular_qualidade_carteira, axis=1)
                    st.dataframe(resumo_co, use_container_width=True)

            with tabs[2]:
                with st.spinner('Desenhando contornos dos clusters...'):
                    st.subheader("Contorno Geográfico dos Clusters")
                    st.write("Este mapa desenha um polígono ao redor de cada hotspot.")
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
                            folium.GeoJson(gdf_hulls, style_function=lambda x: {'color': 'red', 'weight': 2, 'fillColor': 'red', 'fillOpacity': 0.2}, tooltip=folium.GeoJsonTooltip(fields=['cluster', 'contagem', 'area_km2', 'densidade'], aliases=['Cluster ID:', 'Nº de Serviços:', 'Área (km²):', 'Serviços por km²:'], localize=True, sticky=True)).add_to(m_hull)
                            marker_cluster_hull = MarkerCluster().add_to(m_hull)
                            for idx, row in gdf_clusters_reais.iterrows():
                                folium.Marker(location=[row['latitude'], row['longitude']], popup=f"Cluster: {row['cluster']}", icon=folium.Icon(color='blue', icon='info-sign')).add_to(marker_cluster_hull)
                            st_folium(m_hull, use_container_width=True, height=700)
                        except Exception as e:
                            st.warning(f"Não foi possível desenhar os contornos. Erro: {e}")
                    else:
                        st.warning("Nenhum cluster para desenhar.")
            
            if df_metas is not None:
                pacotes_tab_index = 3
                with tabs[pacotes_tab_index]:
                    with st.spinner('Simulando roteirização e desenhando pacotes...'):
                        st.subheader("Simulação de Roteirização Diária")
                        st.write("Este mapa simula a alocação dos serviços agrupados entre as equipes de um CO, respeitando a capacidade de produção de cada uma.")
                        
                        todos_alocados = []
                        todos_excedentes = []
                        for co in gdf_com_clusters['centro_operativo'].unique():
                            gdf_co = gdf_com_clusters[gdf_com_clusters['centro_operativo'] == co].copy()
                            metas_co = df_metas[df_metas['centro_operativo'].str.strip().str.upper() == co.strip().upper()]
                            
                            if not metas_co.empty:
                                n_equipes = int(metas_co['equipes'].iloc[0])
                                capacidade_producao = int(metas_co['produção'].iloc[0])
                                
                                if n_equipes > 0 and capacidade_producao > 0 and len(gdf_co) > 0:
                                    alocados, excedentes = simular_pacotes_por_densidade(gdf_co, n_equipes, capacidade_producao)
                                    todos_alocados.append(alocados)
                                    todos_excedentes.append(excedentes)
                            else: 
                                todos_excedentes.append(gdf_co)

                        gdf_alocados_final = pd.concat(todos_alocados) if todos_alocados else gpd.GeoDataFrame()
                        gdf_excedentes_final = pd.concat(todos_excedentes) if todos_excedentes else gpd.GeoDataFrame()
                        
                        st.markdown("##### Performance da Carteira Agrupada")
                        c1, c2, c3 = st.columns(3)
                        total_servicos_analisados = len(gdf_alocados_final) + len(gdf_excedentes_final)
                        c1.metric("Serviços na Análise", total_servicos_analisados)
                        c2.metric("Serviços Alocados", len(gdf_alocados_final))
                        c3.metric("Serviços Excedentes", len(gdf_excedentes_final), delta=f"-{len(gdf_excedentes_final)} não roteirizados", delta_color="inverse")
                        
                        if not gdf_com_clusters.empty:
                            map_center_pacotes = [gdf_com_clusters.latitude.mean(), gdf_com_clusters.longitude.mean()]
                            m_pacotes = folium.Map(location=map_center_pacotes, zoom_start=10)
                            cores_co = {co: color for co, color in zip(gdf_com_clusters['centro_operativo'].unique(), ['blue', 'green', 'purple', 'orange', 'darkred', 'red', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'lightgreen', 'pink', 'lightblue', 'lightgray', 'black'])}
                            
                            if not gdf_alocados_final.empty:
                                gdf_hulls_pacotes = gdf_alocados_final.dissolve(by=['centro_operativo', 'pacote_id']).convex_hull.reset_index()
                                gdf_hulls_pacotes = gdf_hulls_pacotes.rename(columns={0: 'geometry'}).set_geometry('geometry')
                                
                                counts_pacotes = gdf_alocados_final.groupby(['centro_operativo', 'pacote_id']).size().rename('contagem').reset_index()
                                gdf_hulls_pacotes = gdf_hulls_pacotes.merge(counts_pacotes, on=['centro_operativo', 'pacote_id'])
                                
                                gdf_hulls_pacotes_proj = gdf_hulls_pacotes.to_crs("EPSG:3857")
                                gdf_hulls_pacotes['area_km2'] = (gdf_hulls_pacotes_proj.geometry.area / 1_000_000).round(2)
                                gdf_hulls_pacotes['densidade'] = 0.0
                                non_zero_area = gdf_hulls_pacotes['area_km2'] > 0
                                gdf_hulls_pacotes.loc[non_zero_area, 'densidade'] = (gdf_hulls_pacotes.loc[non_zero_area, 'contagem'] / gdf_hulls_pacotes.loc[non_zero_area, 'area_km2']).round(1)

                                folium.GeoJson(
                                    gdf_hulls_pacotes,
                                    style_function=lambda feature: {
                                        'color': cores_co.get(feature['properties']['centro_operativo'], 'gray'),
                                        'weight': 2.5,
                                        'fillColor': cores_co.get(feature['properties']['centro_operativo'], 'gray'),
                                        'fillOpacity': 0.25
                                    },
                                    tooltip=folium.GeoJsonTooltip(
                                        fields=['centro_operativo', 'pacote_id', 'contagem', 'area_km2', 'densidade'],
                                        aliases=['CO:', 'Pacote:', 'Nº de Serviços:', 'Área (km²):', 'Serviços por km²:'],
                                        localize=True,
                                        sticky=True
                                    )
                                ).add_to(m_pacotes)
                            
                            st_folium(m_pacotes, use_container_width=True, height=700)
                        else:
                            st.info("Nenhum pacote de trabalho para simular.")

            with tabs[-1]:
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown("""
                Esta ferramenta utiliza uma combinação de algoritmos geoespaciais e de aprendizado de máquina para fornecer insights sobre a distribuição de serviços.
                - **Análise de Padrão (NNI - Índice do Vizinho Mais Próximo):** Mede se a distribuição geral dos pontos é agrupada, dispersa ou aleatória. É o primeiro indicador da "saúde" logística da carteira.
                - **Detecção de Hotspots (DBSCAN):** É o coração da análise. O DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é um algoritmo que agrupa pontos que estão densamente próximos, marcando como "ruído" (dispersos) os pontos que estão sozinhos. É ideal para encontrar "bolsões" de serviços sem precisar definir previamente o número de clusters.
                - **Simulação de Pacotes (Ranking de Densidade):** A lógica de negócio para a roteirização prioriza a eficiência. Os hotspots identificados são transformados em "pacotes candidatos". Hotspots muito grandes são subdivididos em pacotes menores que respeitam a capacidade de produção de uma equipe. Todos os candidatos são então ranqueados pela sua densidade (serviços por km²), e os melhores são atribuídos às equipes disponíveis.
                """)
                st.subheader("Perguntas Frequentes (FAQ)")
                st.markdown("""
                - **Por que alguns serviços ficam como "dispersos"?** Um serviço é considerado disperso (ou ruído) pelo DBSCAN se ele não tiver um número mínimo de vizinhos (`Mínimo de Pontos por Cluster`) dentro de um raio de busca (`Raio do Cluster`). Isso indica que ele está geograficamente isolado dos demais.
                - **Como escolher os melhores parâmetros de cluster?** Não há um número mágico. Comece com os padrões (Raio: 1km, Mínimo de Pontos: 20). Se você perceber que muitos serviços que parecem próximos estão como "dispersos", tente aumentar o `Raio do Cluster`. Se clusters muito grandes estão sendo formados, tente diminuir o raio ou aumentar o `Mínimo de Pontos`.
                - **O que significa um pacote excedente na simulação?** Significa que, após atribuir os pacotes mais densos e eficientes para todas as equipes disponíveis, ainda sobraram serviços. Eles podem ser serviços de hotspots de baixa prioridade (baixa densidade) ou serviços já classificados como dispersos.
                """)
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")
else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")
