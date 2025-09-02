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
import h3
from shapely.geometry import Polygon

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
def carregar_dados_otimizado(arquivo_enviado):
    """Lê o arquivo de forma otimizada, carregando apenas as colunas necessárias para a análise."""
    colunas_necessarias = ['latitude', 'longitude', 'sucursal', 'centro_operativo', 'corte_recorte', 'prioridade', 'numero_ordem']
    arquivo_enviado.seek(0)
    
    def processar_dataframe(df):
        df.columns = df.columns.str.lower().str.strip()
        if not all(col in df.columns for col in colunas_necessarias):
            st.error("ERRO: Colunas essenciais (como latitude, longitude, numero_ordem, etc.) não foram encontradas."); st.write("Colunas necessárias:", colunas_necessarias); st.write("Colunas encontradas:", df.columns.tolist())
            return None
        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.').astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.').astype(float)
        df = df.dropna(subset=['latitude', 'longitude'])
        return df

    try:
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
                df_excel_full = pd.read_excel(arquivo_enviado, engine='openpyxl')
                df_excel_full.columns = df_excel_full.columns.str.lower().str.strip()
                colunas_para_usar = [col for col in colunas_necessarias if col in df_excel_full.columns]
                df_excel = df_excel_full[colunas_para_usar]
                st.success("Arquivo Excel lido com sucesso."); return processar_dataframe(df_excel)
            except Exception as e:
                st.error(f"Não foi possível ler o arquivo. Último erro: {e}"); return None

@st.cache_data
def carregar_dados_completos(arquivo_enviado):
    """Lê o arquivo completo com todas as colunas para a funcionalidade de download."""
    arquivo_enviado.seek(0)
    try:
        df = pd.read_csv(arquivo_enviado, encoding='utf-16', sep='\t')
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception:
        try:
            arquivo_enviado.seek(0)
            df = pd.read_csv(arquivo_enviado, encoding='latin-1', sep=None, engine='python')
            df.columns = df.columns.str.lower().str.strip()
            return df
        except Exception:
            try:
                arquivo_enviado.seek(0)
                df = pd.read_excel(arquivo_enviado, engine='openpyxl')
                df.columns = df.columns.str.lower().str.strip()
                return df
            except Exception:
                return None

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
    df_analise = carregar_dados_otimizado(uploaded_file)
    df_original_completo = carregar_dados_completos(uploaded_file)

    if df_analise is not None and df_original_completo is not None:
        st.sidebar.success(f"{len(df_analise)} registros carregados!")
        st.sidebar.markdown("### Filtros da Análise")
        
        filtros = ['sucursal', 'centro_operativo', 'corte_recorte', 'prioridade']
        valores_selecionados = {}
        for coluna in filtros:
            if coluna in df_analise.columns:
                lista_unica = df_analise[coluna].dropna().unique().tolist()
                opcoes = sorted([str(item) for item in lista_unica])
                
                if coluna == 'prioridade':
                    valores_selecionados[coluna] = st.sidebar.multiselect(f"{coluna.replace('_', ' ').title()}", opcoes)
                else:
                    valores_selecionados[coluna] = st.sidebar.selectbox(f"{coluna.replace('_', ' ').title()}", ["Todos"] + opcoes)

        df_filtrado = df_analise.copy()
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
            hex_resolution = st.sidebar.slider("Resolução do Mapa Hexagonal", 5, 10, 8, 1, help="Define o tamanho dos hexágonos. Números maiores = hexágonos menores e mais detalhados.")

            gdf_com_clusters = executar_dbscan(
                gpd.GeoDataFrame(df_filtrado, geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude), crs="EPSG:4326"),
                eps_km=eps_cluster_km, 
                min_samples=min_samples_cluster
            )
            
            st.sidebar.markdown("### 📥 Downloads")
            ordens_agrupadas = gdf_com_clusters[gdf_com_clusters['cluster'] != -1]['numero_ordem']
            ordens_dispersas = gdf_com_clusters[gdf_com_clusters['cluster'] == -1]['numero_ordem']
            df_agrupados_download = df_original_completo[df_original_completo['numero_ordem'].isin(ordens_agrupadas)]
            df_dispersos_download = df_original_completo[df_original_completo['numero_ordem'].isin(ordens_dispersas)]
            csv_agrupados = df_agrupados_download.to_csv(index=False).encode('utf-8-sig')
            csv_dispersos = df_dispersos_download.to_csv(index=False).encode('utf-8-sig')
            st.sidebar.download_button(label="⬇️ Baixar Serviços Agrupados", data=csv_agrupados, file_name='servicos_agrupados.csv', mime='text/csv', disabled=df_agrupados_download.empty)
            st.sidebar.download_button(label="⬇️ Baixar Serviços Dispersos", data=csv_dispersos, file_name='servicos_dispersos.csv', mime='text/csv', disabled=df_dispersos_download.empty)

            tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Análise Geográfica e Mapa", "📊 Resumo por Centro Operativo", "🔥 Mapa Hexagonal de Densidade", "💡 Metodologia"])

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
                }), include_groups=False).reset_index()
                resumo_co['% Agrupados'] = (resumo_co['Nº Agrupados'] / resumo_co['Total de Serviços'] * 100).round(1)
                resumo_co['% Dispersos'] = (resumo_co['Nº Dispersos'] / resumo_co['Total de Serviços'] * 100).round(1)
                resumo_co = resumo_co[['centro_operativo', 'Total de Serviços', 'Nº de Clusters', 'Nº Agrupados', '% Agrupados', 'Nº Dispersos', '% Dispersos']]
                st.dataframe(resumo_co, use_container_width=True)
            
            with tab3:
                st.subheader("Mapa Hexagonal de Densidade")
                st.write("Visualize a densidade de serviços em áreas geográficas fixas. A cor de cada hexágono representa o número de cortes em seu interior. Esta visão é estável e não muda com o zoom.")
                limite_hexbin = 25000
                if len(df_filtrado) <= limite_hexbin:
                    df_filtrado['hex_id'] = df_filtrado.apply(lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], hex_resolution), axis=1)
                    df_hex = df_filtrado.groupby('hex_id').size().reset_index(name='contagem')
                    def hex_to_polygon(hex_id):
                        points = [(lon, lat) for lat, lon in h3.cell_to_boundary(hex_id)]
                        return Polygon(points)
                    df_hex['geometry'] = df_hex['hex_id'].apply(hex_to_polygon)
                    gdf_hex = gpd.GeoDataFrame(df_hex, crs="EPSG:4326")
                    map_center_hex = [df_filtrado.latitude.mean(), df_filtrado.longitude.mean()]
                    m_hex = folium.Map(location=map_center_hex, zoom_start=11)
                    folium.Choropleth(
                        geo_data=gdf_hex.to_json(), data=df_hex, columns=['hex_id', 'contagem'],
                        key_on='feature.properties.hex_id', fill_color='YlOrRd', fill_opacity=0.7,
                        line_opacity=0.2, legend_name='Contagem de Serviços por Hexágono'
                    ).add_to(m_hex)
                    st_folium(m_hex, use_container_width=True, height=700, returned_objects=[])
                else:
                    st.info(f"O mapa hexagonal está desabilitado para seleções com mais de {limite_hexbin:,.0f} pontos para garantir a performance. Por favor, aplique mais filtros para visualizar.".replace(",", "."))

            with tab4:
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown("""
                Para garantir uma análise precisa e confiável, utilizamos duas técnicas complementares da estatística espacial:
                
                #### 1. Clustering Baseado em Densidade (DBSCAN)
                **O que é?** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) é um algoritmo de machine learning que identifica agrupamentos de pontos em um espaço. Ele é a base da nossa contagem de "hotspots".
                
                **Como funciona?** O algoritmo define um "cluster" (ou hotspot) como uma área onde existem muitos pontos próximos uns dos outros. Ele agrupa esses pontos e, crucialmente, identifica os pontos que estão isolados em áreas de baixa densidade, classificando-os como "dispersos" (ou "ruído"). É a partir desta análise que calculamos o Nº de Hotspots, o % de Serviços Agrupados e o % de Serviços Dispersos.
                
                #### 2. Análise do Vizinho Mais Próximo (NNI)
                **O que é?** O NNI (Nearest Neighbor Index) é um índice estatístico que responde a uma pergunta fundamental: "A distribuição dos meus pontos é agrupada, aleatória ou dispersa?" Ele é a base da nossa métrica Padrão de Dispersão.
                
                **Como funciona?** A análise mede a distância média entre cada serviço e seu vizinho mais próximo. Em seguida, compara essa média com a distância que seria esperada se os mesmos serviços estivessem distribuídos de forma perfeitamente aleatória na mesma área geográfica. O resultado é um índice único:
                - **NNI < 1 (Agrupado):** Os serviços estão, em média, mais próximos uns dos outros do que o esperado pelo acaso.
                - **NNI ≈ 1 (Aleatório):** Não há um padrão de distribuição estatisticamente relevante.
                - **NNI > 1 (Disperso):** Os serviços estão, em média, mais espalhados uns dos outros do que o esperado pelo acaso.
                
                Juntas, essas duas técnicas fornecem uma visão completa: o DBSCAN **encontra e conta** os agrupamentos, enquanto o NNI nos dá uma **medida geral** do grau de concentração de toda a sua operação.
                """)
                st.subheader("Perguntas Frequentes (FAQ)")
                st.markdown("""
                #### O agrupamento dos serviços é definido por "círculos"? Os pontos de um "círculo" invadem o outro? Como fica a região aglomerada por 4 "círculos"? Não fica um espaço não mapeado no meio?
                
                Essa é uma ótima pergunta! Ao contrário do que se pode imaginar, o algoritmo DBSCAN não desenha círculos fixos e independentes no mapa. Ele funciona mais como uma "mancha de tinta que se espalha" para identificar as áreas densas.
                
                Pense assim:
                1.  O DBSCAN começa em um ponto.
                2.  Ele verifica se há vizinhos suficientes dentro de um **raio** específico (o "Raio do Cluster (km)" que você ajusta).
                3.  Se houver, ele considera esse ponto parte de um cluster e **se expande** para incluir todos os vizinhos densos, e os vizinhos desses vizinhos, e assim por diante.
                
                Isso significa que:
                - **Não são círculos rígidos:** Os clusters resultantes têm **formas irregulares e orgânicas**, adaptando-se à distribuição real dos seus dados (por exemplo, seguindo o traçado de uma rua ou o contorno de um bairro).
                - **Os agrupamentos se fundem:** Se as "áreas de influência" de pontos próximos se sobrepõem e ambos são densos, eles se tornam parte do **mesmo cluster grande**. Não há "invasão" de círculos, mas sim uma fusão natural.
                - **Não ficam espaços não mapeados no meio:** Em uma região aglomerada por vários pontos densos, o DBSCAN não deixa buracos. Ele forma um único cluster contínuo que cobre toda a área densamente populada por serviços. O resultado é uma representação muito mais fiel das suas "zonas de trabalho" do que simples círculos.
                
                #### Por que o DBSCAN é mais adequado para esta ferramenta do que "mapear por km²" ou "mapas de calor"?
                
                Sua sugestão de "mapear por km²" é excelente e se aproxima muito de uma técnica conhecida como **Análise de Grade** ou **Mapa de Calor Hexagonal** (que adicionamos em uma nova aba!).
                
                Ambas as abordagens são valiosas, mas com focos diferentes:
                - **DBSCAN (Clusters Irregulares):** Ideal para **otimização logística**. Os clusters que ele identifica representam as **"zonas de trabalho naturais"** da sua operação, onde uma equipe pode atender múltiplos serviços com mínimo deslocamento. Ele é focado em *agrupamentos reais de serviços*.
                - **Mapa Hexagonal (Visualização de Densidade em Grade):** Perfeito para **percepção rápida de densidade** e relatórios gerenciais. Ele mostra visualmente onde há maior concentração de pontos em áreas geográficas fixas, independentemente de formarem clusters estatisticamente significativos. É mais focado em *onde está mais "quente" de serviços*.
                
                Para a otimização de rotas e alocação de equipes, os clusters orgânicos do DBSCAN são geralmente mais úteis porque eles delimitam áreas de forma mais inteligente para o campo. O mapa hexagonal, por sua vez, complementa essa visão, mostrando as "manchas" gerais de atividade. Juntos, eles oferecem uma análise completa!
                """)
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")
else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")