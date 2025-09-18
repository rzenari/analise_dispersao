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
import io  # Necessário para o download em Excel
import os
import glob # Para encontrar os arquivos KML

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E TÍTULOS
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análise de Dispersão Geográfica")

st.title("🗺️ Ferramenta de Análise de Dispersão Geográfica")
st.write("Faça o upload da sua planilha de cortes para analisar a distribuição geográfica e identificar clusters")

# ==============================================================================
# 3. FUNÇÕES DE ANÁLISE
# ==============================================================================

@st.cache_data
def carregar_kmls(pasta_projeto):
    """
    Varre a pasta do projeto, encontra todos os arquivos .kml,
    lê, tenta corrigir geometrias inválidas, unifica as válidas
    e retorna um log de depuração.
    """
    kml_files = glob.glob(os.path.join(pasta_projeto, '*.kml'))
    if not kml_files:
        return None, pd.DataFrame([{'Arquivo': 'Nenhum arquivo .kml encontrado', 'Status': 'N/A'}])
    
    debug_log = []
    poligonos_validos = []
    
    for kml_file in kml_files:
        try:
            gdf_kml = gpd.read_file(kml_file, driver='KML')
            gdf_kml = gdf_kml[gdf_kml.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            if gdf_kml.empty:
                debug_log.append({'Arquivo': os.path.basename(kml_file), 'Status': '⚠️ Aviso', 'Erro': 'Nenhum polígono encontrado no arquivo.'})
                continue

            geometrias_corrigidas = []
            for poligono in gdf_kml.geometry:
                if not poligono.is_valid:
                    poligono_corrigido = poligono.buffer(0)
                    if poligono_corrigido.is_valid and not poligono_corrigido.is_empty:
                        geometrias_corrigidas.append(poligono_corrigido)
                else:
                    geometrias_corrigidas.append(poligono)
            
            if geometrias_corrigidas:
                poligonos_validos.extend(geometrias_corrigidas)
                debug_log.append({'Arquivo': os.path.basename(kml_file), 'Status': '✅ Sucesso'})
            else:
                debug_log.append({'Arquivo': os.path.basename(kml_file), 'Status': '❌ Falha', 'Erro': 'Geometria inválida e não pôde ser corrigida.'})

        except Exception as e:
            debug_log.append({'Arquivo': os.path.basename(kml_file), 'Status': '❌ Falha', 'Erro': str(e)})

    if not poligonos_validos:
        return None, pd.DataFrame(debug_log)

    geometria_unificada = gpd.GeoSeries(poligonos_validos).unary_union
    return geometria_unificada, pd.DataFrame(debug_log)

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
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        required_cols = ['centro_operativo', 'equipes', 'serviços_designados', 'produção', 'meta_diária']
        if all(col in df.columns for col in required_cols):
            df['centro_operativo'] = df['centro_operativo'].str.strip().str.upper()
            return df
        else:
            st.error(f"ERRO: A planilha de metas precisa conter as colunas: {', '.join(required_cols)}.")
            return None
    except Exception as e:
        st.error(f"Não foi possível ler a planilha de metas. Erro: {e}")
        return None

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

def simular_pacotes_por_densidade(gdf_co, n_equipes, capacidade_designada):
    """
    Simula pacotes com base na densidade, respeitando estritamente a capacidade de serviços designados.
    """
    if gdf_co.empty or n_equipes == 0 or capacidade_designada == 0:
        return gpd.GeoDataFrame(), gdf_co.copy()

    gdf_clusters_reais = gdf_co.copy()
    pacotes_candidatos = []
    
    for cluster_id in gdf_clusters_reais['cluster'].unique():
        if cluster_id == -1: continue
        gdf_cluster_atual = gdf_clusters_reais[gdf_clusters_reais['cluster'] == cluster_id]
        contagem = len(gdf_cluster_atual)

        if 0 < contagem <= capacidade_designada:
            pacotes_candidatos.append({'indices': gdf_cluster_atual.index, 'pontos': gdf_cluster_atual})
        
        elif contagem > capacidade_designada:
            gdf_temp = gdf_cluster_atual.copy()
            while len(gdf_temp) > 0:
                if len(gdf_temp) <= capacidade_designada:
                    pacotes_candidatos.append({'indices': gdf_temp.index, 'pontos': gdf_temp})
                    break
                
                coords_temp = gdf_temp[['longitude', 'latitude']].values
                tree = cKDTree(coords_temp)
                
                _, indices_vizinhos = tree.query(coords_temp[0], k=min(capacidade_designada, len(coords_temp)))
                
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
                area_km2 = (gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326").to_crs("EPSG:3857").geometry.area / 1_000_000).iloc[0]
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

def calcular_qualidade_carteira(row):
    """Calcula a qualidade da carteira com base nas metas e serviços agrupados."""
    meta_diaria = row.get('meta_diária', 0)
    if pd.isna(meta_diaria) or meta_diaria == 0: return "Sem Meta"
    n_agrupados = row.get('Agrupado', 0)
    total_servicos = row.get('total', 0)
    
    if n_agrupados >= meta_diaria: return "✅ Ótima"
    elif total_servicos >= meta_diaria: return "⚠️ Atenção"
    else: return "❌ Crítica"

# ==============================================================================
# 4. LÓGICA PRINCIPAL DA APLICAÇÃO
# ==============================================================================
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("1. Escolha a planilha de cortes", type=["csv", "xlsx", "xls"])
metas_file = st.sidebar.file_uploader("2. Escolha a planilha de metas (Opcional)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df_completo_original = carregar_dados_completos(uploaded_file)
    df_metas = carregar_dados_metas(metas_file)
    
    if df_completo_original is not None:
        st.sidebar.success(f"{len(df_completo_original)} registros carregados!")
        
        kml_polygons, kml_debug_log = carregar_kmls('.')
        if not kml_debug_log.empty:
            sucesso_count = (kml_debug_log['Status'] == '✅ Sucesso').sum()
            st.sidebar.info(f"{sucesso_count} arquivo(s) KML carregado(s) com sucesso.")
            with st.sidebar.expander("🔍 Depurador de Arquivos KML"):
                st.dataframe(kml_debug_log)
        
        if df_metas is not None: 
            st.sidebar.info(f"Metas carregadas para {len(df_metas)} COs.")

        st.sidebar.markdown("### Filtros da Análise")
        filtros = ['sucursal', 'centro_operativo', 'corte_recorte', 'prioridade']
        valores_selecionados = {}
        for coluna in filtros:
            if coluna in df_completo_original.columns:
                lista_unica = df_completo_original[coluna].dropna().unique().tolist()
                opcoes = sorted([str(item) for item in lista_unica])
                if coluna == 'prioridade':
                    valores_selecionados[coluna] = st.sidebar.multiselect(f"{coluna.replace('_', ' ').title()}", opcoes)
                else:
                    valores_selecionados[coluna] = st.sidebar.selectbox(f"{coluna.replace('_', ' ').title()}", ["Todos"] + opcoes)

        df_filtrado = df_completo_original.copy()
        for coluna, valor in valores_selecionados.items():
            if coluna in df_filtrado.columns:
                if coluna == 'prioridade':
                    if valor: df_filtrado = df_filtrado[df_filtrado[coluna].astype(str).isin(valor)]
                elif valor != "Todos": df_filtrado = df_filtrado[df_filtrado[coluna].astype(str) == valor]

        gdf_filtrado_base = gpd.GeoDataFrame(
            df_filtrado, 
            geometry=gpd.points_from_xy(df_filtrado.longitude, df_filtrado.latitude), 
            crs="EPSG:4326"
        )

        gdf_filtrado_base['classificacao'] = 'A ser definido'
        gdf_risco = gpd.GeoDataFrame()

        if kml_polygons is not None:
            indices_risco = gdf_filtrado_base.within(kml_polygons)
            gdf_filtrado_base.loc[indices_risco, 'classificacao'] = 'Área de Risco'
            gdf_risco = gdf_filtrado_base[indices_risco].copy()

        gdf_para_analise = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'A ser definido'].copy()
        
        st.sidebar.markdown("### Parâmetros de Cluster")
        eps_cluster_km = st.sidebar.slider("Raio do Cluster (km)", 0.1, 5.0, 1.0, 0.1)
        min_samples_cluster = st.sidebar.slider("Mínimo de Pontos por Cluster", 2, 20, 20, 1)

        if not gdf_para_analise.empty:
            gdf_com_clusters = executar_dbscan(gdf_para_analise, eps_km=eps_cluster_km, min_samples=min_samples_cluster)
            gdf_filtrado_base.loc[gdf_com_clusters[gdf_com_clusters['cluster'] != -1].index, 'classificacao'] = 'Agrupado'
            gdf_filtrado_base.loc[gdf_com_clusters[gdf_com_clusters['cluster'] == -1].index, 'classificacao'] = 'Disperso'
            gdf_filtrado_base = gdf_filtrado_base.merge(gdf_com_clusters[['cluster']], left_index=True, right_index=True, how='left')
        else:
             gdf_filtrado_base['cluster'] = -1
        
        gdf_filtrado_base['cluster'] = gdf_filtrado_base['cluster'].fillna(-1)
        gdf_filtrado_base.loc[gdf_filtrado_base['classificacao'] == 'A ser definido', 'classificacao'] = 'Disperso'

        st.header("Resultados da Análise")
        
        if not gdf_filtrado_base.empty:
            st.sidebar.markdown("### Filtro de Visualização do Mapa")
            opcoes_visualizacao = ["Todos", "Agrupado", "Disperso"]
            if not gdf_risco.empty:
                opcoes_visualizacao.append("Área de Risco")
            tipo_visualizacao = st.sidebar.radio("Mostrar nos mapas:", opcoes_visualizacao)
            
            st.sidebar.markdown("### 📥 Downloads")
            
            df_agrupados_download = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'].drop(columns=['geometry', 'cluster'], errors='ignore')
            if not df_agrupados_download.empty:
                csv_agrupados = df_agrupados_download.to_csv(index=False).encode('utf-8-sig')
                st.sidebar.download_button(label="⬇️ Baixar Agrupados (CSV)", data=csv_agrupados, file_name='servicos_agrupados.csv', mime='text/csv')
            
            df_dispersos_download = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Disperso'].drop(columns=['geometry', 'cluster'], errors='ignore')
            if not df_dispersos_download.empty:
                csv_dispersos = df_dispersos_download.to_csv(index=False).encode('utf-8-sig')
                st.sidebar.download_button(label="⬇️ Baixar Dispersos (CSV)", data=csv_dispersos, file_name='servicos_dispersos.csv', mime='text/csv')

            if not gdf_risco.empty:
                df_risco_download = gdf_risco.drop(columns=['geometry', 'cluster'], errors='ignore')
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_risco_download.to_excel(writer, index=False, sheet_name='Area_de_Risco')
                excel_data = output.getvalue()
                st.sidebar.download_button(
                    label="⬇️ Baixar Área de Risco (Excel)",
                    data=excel_data,
                    file_name='servicos_area_risco.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            gdf_visualizacao = gdf_filtrado_base.copy()
            if tipo_visualizacao != "Todos":
                 gdf_visualizacao = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == tipo_visualizacao]

            lista_abas = ["🗺️ Análise Geográfica", "📊 Resumo por CO", "📍 Contorno dos Clusters"]
            if df_metas is not None: lista_abas.append("📦 Pacotes de Trabalho")
            lista_abas.append("💡 Metodologia")
            tabs = st.tabs(lista_abas)

            with tabs[0]:
                with st.spinner('Carregando análise e mapa...'):
                    st.subheader("Resumo da Análise de Classificação")
                    
                    total_servicos = len(gdf_filtrado_base)
                    n_agrupados = len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'])
                    n_dispersos = len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Disperso'])
                    n_risco = len(gdf_risco)

                    p_agrupados = (n_agrupados / total_servicos * 100) if total_servicos > 0 else 0
                    p_dispersos = (n_dispersos / total_servicos * 100) if total_servicos > 0 else 0
                    p_risco = (n_risco / total_servicos * 100) if total_servicos > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Nº Agrupados", f"{n_agrupados}", f"{p_agrupados:.1f}%")
                    col2.metric("Nº Dispersos", f"{n_dispersos}", f"{p_dispersos:.1f}%")
                    col3.metric("Nº em Área de Risco", f"{n_risco}", f"{p_risco:.1f}%")

                    st.subheader(f"Mapa Interativo")
                    st.write("Serviços em azul são Agrupados, cinza são Dispersos e vermelho estão em Área de Risco.")
                    if not gdf_visualizacao.empty:
                        map_center = [gdf_visualizacao.latitude.mean(), gdf_visualizacao.longitude.mean()]
                        m = folium.Map(location=map_center, zoom_start=11)
                        
                        if kml_polygons is not None:
                            folium.GeoJson(kml_polygons, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.1}, tooltip="Área de Risco / Ilha").add_to(m)

                        for _, row in gdf_visualizacao.iterrows():
                            cor_classificacao = {'Agrupado': 'blue', 'Disperso': 'gray', 'Área de Risco': 'red'}
                            folium.CircleMarker(
                                location=[row['latitude'], row['longitude']],
                                radius=5,
                                color=cor_classificacao.get(row['classificacao'], 'black'),
                                fill=True,
                                fill_color=cor_classificacao.get(row['classificacao'], 'black'),
                                fill_opacity=0.7,
                                popup=f"Classificação: {row['classificacao']}"
                            ).add_to(m)
                        st_folium(m, use_container_width=True, height=700)
                    else:
                        st.warning("Nenhum serviço para exibir no mapa com os filtros atuais.")

            with tabs[1]:
                with st.spinner('Gerando tabela de resumo...'):
                    st.subheader("Resumo por Centro Operativo")
                    
                    resumo_co = gdf_filtrado_base.groupby('centro_operativo')['classificacao'].value_counts().unstack(fill_value=0)
                    
                    for col in ['Agrupado', 'Disperso', 'Área de Risco']:
                        if col not in resumo_co.columns:
                            resumo_co[col] = 0
                    
                    resumo_co['total'] = resumo_co['Agrupado'] + resumo_co['Disperso'] + resumo_co['Área de Risco']
                    resumo_co['% Agrupado'] = (resumo_co['Agrupado'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Disperso'] = (resumo_co['Disperso'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Área de Risco'] = (resumo_co['Área de Risco'] / resumo_co['total'] * 100).round(1)
                    resumo_co.reset_index(inplace=True)

                    if df_metas is not None:
                        resumo_co['centro_operativo_join_key'] = resumo_co['centro_operativo'].str.strip().str.upper()
                        df_metas['centro_operativo_join_key'] = df_metas['centro_operativo'].str.strip().str.upper()
                        resumo_co = pd.merge(resumo_co, df_metas, on='centro_operativo_join_key', how='left')
                        
                        if 'centro_operativo_x' in resumo_co.columns:
                           resumo_co = resumo_co.drop(columns=['centro_operativo_y', 'centro_operativo_join_key']).rename(columns={'centro_operativo_x': 'centro_operativo'})
                        
                        resumo_co['qualidade_da_carteira'] = resumo_co.apply(calcular_qualidade_carteira, axis=1)
                        cols_ordem = ['centro_operativo', 'total', 'Agrupado', '% Agrupado', 'Disperso', '% Disperso', 'Área de Risco', '% Área de Risco', 'qualidade_da_carteira']
                        cols_existentes = [col for col in cols_ordem if col in resumo_co.columns]
                        resumo_co = resumo_co[cols_existentes]

                    st.dataframe(resumo_co, use_container_width=True)

            with tabs[2]:
                with st.spinner('Desenhando contornos dos clusters...'):
                    st.subheader("Contorno Geográfico dos Clusters (Hotspots)")
                    st.write("Este mapa desenha um polígono ao redor de cada hotspot da categoria 'Agrupado'.")
                    
                    gdf_clusters_reais = gdf_filtrado_base[(gdf_filtrado_base['classificacao'] == 'Agrupado') & (gdf_filtrado_base['cluster'] != -1)]
                    if not gdf_clusters_reais.empty:
                        map_center_hull = [gdf_clusters_reais.latitude.mean(), gdf_clusters_reais.longitude.mean()]
                        m_hull = folium.Map(location=map_center_hull, zoom_start=11)
                        if kml_polygons is not None:
                            folium.GeoJson(kml_polygons, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.1}).add_to(m_hull)
                        try:
                            hulls = gdf_clusters_reais.dissolve(by='cluster', aggfunc={'centro_operativo': 'first'}).convex_hull
                            gdf_hulls = gpd.GeoDataFrame(geometry=hulls).reset_index()
                            folium.GeoJson(gdf_hulls, style_function=lambda x: {'color': 'blue', 'weight': 2.5, 'fillColor': 'blue', 'fillOpacity': 0.2}, tooltip=folium.GeoJsonTooltip(fields=['cluster'], aliases=['Hotspot ID:'])).add_to(m_hull)
                            st_folium(m_hull, use_container_width=True, height=700)
                        except Exception as e:
                            st.warning(f"Não foi possível desenhar os contornos. Erro: {e}")
                    else:
                        st.warning("Nenhum cluster para desenhar.")

            if df_metas is not None:
                pacotes_tab_index = 3
                with tabs[pacotes_tab_index]:
                    with st.spinner('Simulando roteirização e desenhando pacotes...'):
                        
                        todos_alocados = []
                        todos_excedentes = []
                        
                        gdf_para_pacotes = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'].copy()
                        cos_filtrados = gdf_para_pacotes['centro_operativo'].unique() if not gdf_para_pacotes.empty else []
                        
                        for co in cos_filtrados:
                            gdf_co = gdf_para_pacotes[gdf_para_pacotes['centro_operativo'] == co].copy()
                            metas_co = df_metas[df_metas['centro_operativo'].str.strip().str.upper() == co.strip().upper()]
                            
                            if not metas_co.empty:
                                n_equipes = int(metas_co['equipes'].iloc[0])
                                capacidade_designada = int(metas_co['serviços_designados'].iloc[0])
                                
                                if n_equipes > 0 and capacidade_designada > 0 and len(gdf_co) > 0:
                                    alocados, excedentes_co = simular_pacotes_por_densidade(gdf_co, n_equipes, capacidade_designada)
                                    todos_alocados.append(alocados)
                                    todos_excedentes.append(excedentes_co)
                            else: 
                                todos_excedentes.append(gdf_co)
                        
                        gdf_servicos_restantes = gdf_filtrado_base[gdf_filtrado_base['classificacao'] != 'Agrupado'].copy()
                        if not gdf_servicos_restantes.empty:
                            todos_excedentes.append(gdf_servicos_restantes)

                        gdf_alocados_final = pd.concat(todos_alocados, ignore_index=True) if todos_alocados else gpd.GeoDataFrame()
                        gdf_excedentes_final = pd.concat(todos_excedentes, ignore_index=True) if todos_excedentes else gpd.GeoDataFrame()
                        
                        if not gdf_alocados_final.empty:
                            df_pacotes_download = gdf_alocados_final.drop(columns=['geometry', 'cluster'], errors='ignore')
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_pacotes_download.to_excel(writer, index=False, sheet_name='Pacotes_Alocados')
                            excel_data = output.getvalue()
                            st.sidebar.download_button(
                                label="⬇️ Baixar Pacotes de Trabalho (Excel)",
                                data=excel_data,
                                file_name='pacotes_de_trabalho.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                            )

                        st.subheader("Painel de Simulação")
                        metas_filtradas = df_metas[df_metas['centro_operativo'].isin(cos_filtrados)]
                        
                        if not metas_filtradas.empty:
                            equipes_disponiveis = metas_filtradas['equipes'].sum()
                            meta_diaria_total = metas_filtradas['meta_diária'].sum()
                            metas_filtradas['expectativa_execucao'] = metas_filtradas['equipes'] * metas_filtradas['produção']
                            expectativa_total = metas_filtradas['expectativa_execucao'].sum()
                            
                            servicos_agrupados_para_pacotes = len(gdf_para_pacotes)
                            servicos_alocados = len(gdf_alocados_final)
                            pacotes_criados = gdf_alocados_final['pacote_id'].nunique() if not gdf_alocados_final.empty else 0
                            servicos_excedentes = len(gdf_excedentes_final)

                            aderencia_meta = (servicos_alocados / meta_diaria_total * 100) if meta_diaria_total > 0 else 0
                            ocupacao_equipes = (pacotes_criados / equipes_disponiveis * 100) if equipes_disponiveis > 0 else 0

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("##### Parâmetros de Planejamento")
                                st.metric("Equipes Disponíveis", f"{int(equipes_disponiveis)}")
                                st.metric("Meta Diária (CO)", f"{int(meta_diaria_total)}")
                                st.metric("Expectativa de Execução", f"{int(expectativa_total)}")
                            with col2:
                                st.markdown("##### Resultado da Simulação")
                                st.metric("Serviços Agrupados (Roteirizáveis)", f"{servicos_agrupados_para_pacotes}")
                                st.metric("Serviços Alocados", f"{servicos_alocados}")
                                st.metric("Pacotes Criados", f"{pacotes_criados}")
                                st.metric("Serviços Excedentes", f"{servicos_excedentes}")
                            with col3:
                                st.markdown("##### Análise de Desempenho")
                                st.metric("Aderência à Meta", f"{aderencia_meta:.1f}%")
                                st.metric("Ocupação das Equipes", f"{ocupacao_equipes:.1f}%")
                        
                        st.markdown("---")
                        
                        if not gdf_filtrado_base.empty:
                            map_center_pacotes = [gdf_filtrado_base.latitude.mean(), gdf_filtrado_base.longitude.mean()]
                            m_pacotes = folium.Map(location=map_center_pacotes, zoom_start=10)
                            cores_co = {co: color for co, color in zip(gdf_filtrado_base['centro_operativo'].unique(), ['blue', 'green', 'purple', 'orange', 'darkred', 'red', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'lightgreen', 'pink', 'lightblue', 'lightgray', 'black'])}
                            if kml_polygons is not None:
                                folium.GeoJson(kml_polygons, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.1}).add_to(m_pacotes)

                            if not gdf_alocados_final.empty:
                                gdf_hulls_pacotes = gdf_alocados_final.dissolve(by=['centro_operativo', 'pacote_id']).convex_hull.reset_index()
                                gdf_hulls_pacotes = gdf_hulls_pacotes.rename(columns={0: 'geometry'}).set_geometry('geometry')
                                
                                counts_pacotes = gdf_alocados_final.groupby(['centro_operativo', 'pacote_id']).size().rename('contagem').reset_index()
                                gdf_hulls_pacotes = gdf_hulls_pacotes.merge(counts_pacotes, on=['centro_operativo', 'pacote_id'])
                                
                                gdf_hulls_pacotes_proj = gdf_hulls_pacotes.to_crs("EPSG:3857")
                                gdf_hulls_pacotes['area_km2'] = (gdf_hulls_pacotes_proj.geometry.area / 1_000_000).round(2)
                                
                                folium.GeoJson(
                                    gdf_hulls_pacotes,
                                    style_function=lambda feature: {'color': cores_co.get(feature['properties']['centro_operativo'], 'gray'), 'weight': 2.5, 'fillColor': cores_co.get(feature['properties']['centro_operativo'], 'gray'), 'fillOpacity': 0.25},
                                    tooltip=folium.GeoJsonTooltip(fields=['centro_operativo', 'pacote_id', 'contagem', 'area_km2'], aliases=['CO:', 'Pacote:', 'Nº de Serviços:', 'Área (km²):'], localize=True, sticky=True)
                                ).add_to(m_pacotes)
                            
                            st_folium(m_pacotes, use_container_width=True, height=700)
                        else:
                            st.info("Nenhum pacote de trabalho para simular.")
            with tabs[-1]:
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown(""" (O conteúdo da metodologia será inserido aqui) """)
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")
else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")
