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
import io
import os
import glob
import zipfile
import requests
from datetime import datetime

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
    Varre a pasta, lê arquivos .kml e .kmz, corrige geometrias inválidas
    e retorna um dicionário de geometrias individuais e um log de depuração.
    """
    kml_files = glob.glob(os.path.join(pasta_projeto, '*.kml'))
    kmz_files = glob.glob(os.path.join(pasta_projeto, '*.kmz'))
    all_gis_files = kml_files + kmz_files

    if not all_gis_files:
        return None, pd.DataFrame([{'Arquivo': 'Nenhum arquivo .kml ou .kmz encontrado', 'Status': 'N/A'}])
    
    debug_log = []
    geometrias_individuais = {}
    
    for gis_file in all_gis_files:
        nome_arquivo = os.path.basename(gis_file)
        try:
            gdf_file = None
            if gis_file.lower().endswith('.kml'):
                gdf_file = gpd.read_file(gis_file, driver='KML')
            elif gis_file.lower().endswith('.kmz'):
                with zipfile.ZipFile(gis_file, 'r') as z:
                    kml_filename = next((name for name in z.namelist() if name.lower().endswith('.kml')), None)
                    if kml_filename:
                        with z.open(kml_filename) as kml_content:
                            gdf_file = gpd.read_file(kml_content, driver='KML')
            
            if gdf_file is None or gdf_file.empty:
                debug_log.append({'Arquivo': nome_arquivo, 'Status': '⚠️ Aviso', 'Erro': 'Nenhum polígono encontrado.'})
                continue

            gdf_file = gdf_file[gdf_file.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            geometrias_corrigidas = []
            for poligono in gdf_file.geometry:
                if not poligono.is_valid:
                    poligono_corrigido = poligono.buffer(0)
                    if poligono_corrigido.is_valid and not poligono_corrigido.is_empty:
                        geometrias_corrigidas.append(poligono_corrigido)
                else:
                    geometrias_corrigidas.append(poligono)
            
            if geometrias_corrigidas:
                geometrias_individuais[nome_arquivo] = gpd.GeoSeries(geometrias_corrigidas).unary_union
                debug_log.append({'Arquivo': nome_arquivo, 'Status': '✅ Sucesso'})
            else:
                debug_log.append({'Arquivo': nome_arquivo, 'Status': '❌ Falha', 'Erro': 'Geometria inválida, não foi possível corrigir.'})

        except Exception as e:
            debug_log.append({'Arquivo': nome_arquivo, 'Status': '❌ Falha', 'Erro': str(e)})

    if not geometrias_individuais:
        return None, pd.DataFrame(debug_log)

    return geometrias_individuais, pd.DataFrame(debug_log)


def carregar_dados_completos(arquivo_enviado):
    """Lê o arquivo completo com todas as colunas."""
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

def df_to_excel(df):
    """Converte um DataFrame para um objeto BytesIO em formato Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')
    return output.getvalue()

@st.cache_data(ttl=10800)
def get_weather_forecast(lat, lon, api_key):
    """Busca a previsão de 5 dias da API OpenWeatherMap."""
    URL = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=pt_br"
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        daily_forecasts = {}
        for forecast in data['list']:
            date = datetime.fromtimestamp(forecast['dt']).strftime('%Y-%m-%d')
            # Prioriza a previsão do meio-dia para representar o dia
            if date not in daily_forecasts or forecast['dt_txt'].endswith('12:00:00'):
                daily_forecasts[date] = forecast
        
        forecast_list = []
        for date, forecast in sorted(daily_forecasts.items())[:5]:
            forecast_list.append({
                "date": datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m'),
                "condition": forecast['weather'][0]['description'].title(),
                "icon": f"https://openweathermap.org/img/wn/{forecast['weather'][0]['icon']}@2x.png",
                "wind_speed_kmh": round(forecast['wind']['speed'] * 3.6, 1)
            })
        return forecast_list
    except requests.exceptions.RequestException as e:
        return f"Erro ao buscar dados: {e}"

def get_operational_status(condition, wind_speed):
    """Define o status da operação com base no clima."""
    is_rainy = any(keyword in condition.lower() for keyword in ["chuva", "tempestade", "chuvisco"])
    is_windy = wind_speed > 40.0
    
    if is_rainy or is_windy:
        return "⚠️ Possível Contingência"
    else:
        return "✅ Operação Normal"
        
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
        
        geometrias_kml_dict, kml_debug_log = carregar_kmls('.')
        if kml_debug_log is not None:
            sucesso_count = (kml_debug_log['Status'] == '✅ Sucesso').sum()
            st.sidebar.info(f"{sucesso_count} arquivo(s) KML/KMZ carregado(s) com sucesso.")
            with st.sidebar.expander("🔍 Depurador de Arquivos KML/KMZ"):
                st.dataframe(kml_debug_log)
        
        if df_metas is not None: 
            st.sidebar.info(f"Metas carregadas para {len(df_metas)} COs.")

        areas_sem_laranja = []
        if geometrias_kml_dict:
            st.sidebar.markdown("### Controle da Área Laranja")
            nomes_areas = list(geometrias_kml_dict.keys())
            areas_sem_laranja = st.sidebar.multiselect('Desativar Área Laranja para:', nomes_areas, help="Selecione as áreas de risco que NÃO devem ter a área laranja de 120m ao redor.")

        kml_risco_unificado, kml_laranja_unificado = None, None
        if geometrias_kml_dict:
            kml_risco_unificado = gpd.GeoSeries(list(geometrias_kml_dict.values()), crs="EPSG:4326").unary_union
            
            poligonos_para_buffer = [poly for name, poly in geometrias_kml_dict.items() if name not in areas_sem_laranja]
            if poligonos_para_buffer:
                geometria_para_buffer = gpd.GeoSeries(poligonos_para_buffer, crs="EPSG:4326").unary_union
                geometria_proj = gpd.GeoSeries([geometria_para_buffer], crs="EPSG:4326").to_crs("EPSG:3857")
                buffer_grande = geometria_proj.buffer(120)
                geometria_laranja_proj = buffer_grande.difference(geometria_proj)
                kml_laranja_unificado = gpd.GeoSeries(geometria_laranja_proj, crs="EPSG:3857").to_crs("EPSG:4326").unary_union

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
        gdf_risco, gdf_laranja = gpd.GeoDataFrame(), gpd.GeoDataFrame()

        if kml_risco_unificado is not None:
            indices_risco = gdf_filtrado_base.within(kml_risco_unificado)
            gdf_filtrado_base.loc[indices_risco, 'classificacao'] = 'Área de Risco'
            gdf_risco = gdf_filtrado_base[indices_risco].copy()

        if kml_laranja_unificado is not None and not kml_laranja_unificado.is_empty:
            gdf_temp_para_laranja = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'A ser definido']
            indices_laranja = gdf_temp_para_laranja.within(kml_laranja_unificado)
            gdf_filtrado_base.loc[indices_laranja[indices_laranja].index, 'classificacao'] = 'Área Laranja'
            gdf_laranja = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Área Laranja'].copy()

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

        gdf_alocados_final = gpd.GeoDataFrame()
        if df_metas is not None:
            todos_alocados, todos_excedentes = [], []
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
        
        st.header("Resultados da Análise")
        
        if not gdf_filtrado_base.empty:
            st.sidebar.markdown("### Filtro de Visualização do Mapa")
            opcoes_visualizacao = ["Todos", "Agrupado", "Disperso", "Área de Risco", "Área Laranja"]
            tipo_visualizacao = st.sidebar.radio("Mostrar nos mapas:", opcoes_visualizacao)
            
            st.sidebar.markdown("### 📥 Downloads")
            
            df_agrupados_download = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'].drop(columns=['geometry', 'cluster'], errors='ignore')
            if not df_agrupados_download.empty:
                st.sidebar.download_button(label="⬇️ Baixar Agrupados (Excel)", data=df_to_excel(df_agrupados_download), file_name='servicos_agrupados.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            
            df_dispersos_download = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Disperso'].drop(columns=['geometry', 'cluster'], errors='ignore')
            if not df_dispersos_download.empty:
                st.sidebar.download_button(label="⬇️ Baixar Dispersos (Excel)", data=df_to_excel(df_dispersos_download), file_name='servicos_dispersos.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

            if not gdf_risco.empty:
                st.sidebar.download_button(label="⬇️ Baixar Área de Risco (Excel)", data=df_to_excel(gdf_risco), file_name='servicos_area_risco.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            
            if not gdf_laranja.empty:
                st.sidebar.download_button(label="⬇️ Baixar Área Laranja (Excel)", data=df_to_excel(gdf_laranja), file_name='servicos_area_laranja.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            
            if not gdf_alocados_final.empty:
                st.sidebar.download_button(
                    label="⬇️ Baixar Pacotes de Trabalho (Excel)",
                    data=df_to_excel(gdf_alocados_final.drop(columns=['geometry', 'cluster'], errors='ignore')),
                    file_name='pacotes_de_trabalho.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            gdf_visualizacao = gdf_filtrado_base.copy()
            if tipo_visualizacao != "Todos":
                 gdf_visualizacao = gdf_filtrado_base[gdf_filtrado_base['classificacao'] == tipo_visualizacao]

            lista_abas = ["🗺️ Análise Geográfica", "📊 Resumo por CO", "📍 Contorno dos Clusters"]
            if df_metas is not None:
                lista_abas.append("📦 Pacotes de Trabalho")
            lista_abas.append("🌦️ Previsão do Tempo")
            lista_abas.append("💡 Metodologia")
            tabs = st.tabs(lista_abas)
            
            tab_index = 0

            with tabs[tab_index]: # Análise Geográfica
                with st.spinner('Carregando análise e mapa...'):
                    st.subheader("Resumo da Análise de Classificação")
                    
                    total_servicos = len(gdf_filtrado_base)
                    n_agrupados = len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'])
                    n_dispersos = len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Disperso'])
                    n_risco = len(gdf_risco)
                    n_laranja = len(gdf_laranja)
                    
                    p_agrupados = (n_agrupados / total_servicos * 100) if total_servicos > 0 else 0
                    p_dispersos = (n_dispersos / total_servicos * 100) if total_servicos > 0 else 0
                    p_risco = (n_risco / total_servicos * 100) if total_servicos > 0 else 0
                    p_laranja = (n_laranja / total_servicos * 100) if total_servicos > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Nº Agrupados", f"{n_agrupados}", f"{p_agrupados:.1f}%")
                    col2.metric("Nº Dispersos", f"{n_dispersos}", f"{p_dispersos:.1f}%")
                    col3.metric("Nº em Área de Risco", f"{n_risco}", f"{p_risco:.1f}%")
                    col4.metric("Nº em Área Laranja", f"{n_laranja}", f"{p_laranja:.1f}%")

                    st.subheader(f"Mapa Interativo")
                    if not gdf_visualizacao.empty:
                        map_center = [gdf_visualizacao.latitude.mean(), gdf_visualizacao.longitude.mean()]
                        m = folium.Map(location=map_center, zoom_start=11)
                        if geometrias_kml_dict:
                            for nome, poligono in geometrias_kml_dict.items():
                                folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.2}, tooltip=f"Área de Risco: {nome}").add_to(m)
                        
                        if kml_laranja_dict:
                            for nome, poligono in kml_laranja_dict.items():
                                if poligono and not poligono.is_empty:
                                    folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'orange', 'color': 'orange', 'weight': 2, 'fillOpacity': 0.2}, tooltip=f"Área Laranja: {nome}").add_to(m)
                        
                        cor_classificacao = {'Agrupado': 'blue', 'Disperso': 'gray', 'Área de Risco': 'red', 'Área Laranja': 'orange'}
                        for _, row in gdf_visualizacao.iterrows():
                            folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=5, color=cor_classificacao.get(row['classificacao'], 'black'), fill=True, fill_color=cor_classificacao.get(row['classificacao'], 'black'), fill_opacity=0.7, popup=f"Classificação: {row['classificacao']}").add_to(m)
                        st_folium(m, use_container_width=True, height=700)
                    else:
                        st.warning("Nenhum serviço para exibir no mapa com os filtros atuais.")
            tab_index += 1

            with tabs[tab_index]: # Resumo por CO
                with st.spinner('Gerando tabela de resumo...'):
                    st.subheader("Resumo por Centro Operativo")
                    
                    resumo_co = gdf_filtrado_base.groupby('centro_operativo')['classificacao'].value_counts().unstack(fill_value=0)
                    for col in ['Agrupado', 'Disperso', 'Área de Risco', 'Área Laranja']:
                        if col not in resumo_co.columns: resumo_co[col] = 0
                    
                    resumo_co['total'] = resumo_co['Agrupado'] + resumo_co['Disperso'] + resumo_co['Área de Risco'] + resumo_co['Área Laranja']
                    resumo_co['% Agrupado'] = (resumo_co['Agrupado'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Disperso'] = (resumo_co['Disperso'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Área de Risco'] = (resumo_co['Área de Risco'] / resumo_co['total'] * 100).round(1)
                    resumo_co['% Área Laranja'] = (resumo_co['Área Laranja'] / resumo_co['total'] * 100).round(1)
                    resumo_co.reset_index(inplace=True)

                    if df_metas is not None:
                        resumo_simulacao = gdf_alocados_final.groupby('centro_operativo').agg(
                            Serviços_Alocados=('pacote_id', 'size'),
                            Pacotes_Criados=('pacote_id', 'nunique')
                        ).reset_index()
                        resumo_co = pd.merge(resumo_co, resumo_simulacao, on='centro_operativo', how='left')
                        
                        resumo_co['centro_operativo_join_key'] = resumo_co['centro_operativo'].str.strip().str.upper()
                        df_metas['centro_operativo_join_key'] = df_metas['centro_operativo'].str.strip().str.upper()
                        resumo_co = pd.merge(resumo_co, df_metas, on='centro_operativo_join_key', how='left')
                        
                        if 'centro_operativo_x' in resumo_co.columns:
                           resumo_co = resumo_co.drop(columns=['centro_operativo_y', 'centro_operativo_join_key']).rename(columns={'centro_operativo_x': 'centro_operativo'})
                        
                        resumo_co['Expectativa_Execução'] = resumo_co['equipes'] * resumo_co['produção']
                        resumo_co['Aderência_à_Meta_%'] = (resumo_co['Serviços_Alocados'] / resumo_co['meta_diária'] * 100).fillna(0).round(1)
                        resumo_co['Ocupação_das_Equipes_%'] = (resumo_co['Pacotes_Criados'] / resumo_co['equipes'] * 100).fillna(0).round(1)
                        resumo_co['qualidade_da_carteira'] = resumo_co.apply(calcular_qualidade_carteira, axis=1)
                        
                        cols_ordem = ['centro_operativo', 'total', 'Agrupado', '% Agrupado', 'Disperso', '% Disperso', 'Área de Risco', '% Área de Risco', 'Área Laranja', '% Área Laranja', 'equipes', 'meta_diária', 'Expectativa_Execução', 'Serviços_Alocados', 'Pacotes_Criados', 'Aderência_à_Meta_%', 'Ocupação_das_Equipes_%', 'qualidade_da_carteira']
                        cols_existentes = [col for col in cols_ordem if col in resumo_co.columns]
                        resumo_co = resumo_co[cols_existentes].fillna(0)

                    st.dataframe(resumo_co, use_container_width=True)
            tab_index += 1

            with tabs[tab_index]: # Contorno dos Clusters
                with st.spinner('Desenhando contornos dos clusters...'):
                    st.subheader("Contorno Geográfico dos Clusters (Hotspots)")
                    st.write("Este mapa desenha um polígono ao redor de cada hotspot da categoria 'Agrupado'.")
                    
                    gdf_clusters_reais = gdf_filtrado_base[(gdf_filtrado_base['classificacao'] == 'Agrupado') & (gdf_filtrado_base['cluster'] != -1)]
                    if not gdf_clusters_reais.empty:
                        map_center_hull = [gdf_clusters_reais.latitude.mean(), gdf_clusters_reais.longitude.mean()]
                        m_hull = folium.Map(location=map_center_hull, zoom_start=11)
                        if geometrias_kml_dict:
                            for nome, poligono in geometrias_kml_dict.items():
                                folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.1}, tooltip=f"Área de Risco: {nome}").add_to(m_hull)
                        if kml_laranja_dict:
                            for nome, poligono in kml_laranja_dict.items():
                                if poligono and not poligono.is_empty:
                                    folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'orange', 'color': 'orange', 'weight': 2, 'fillOpacity': 0.1}, tooltip=f"Área Laranja: {nome}").add_to(m_hull)
                        try:
                            hulls = gdf_clusters_reais.dissolve(by='cluster').convex_hull
                            gdf_hulls = gpd.GeoDataFrame(geometry=hulls).reset_index()
                            folium.GeoJson(gdf_hulls, style_function=lambda x: {'color': 'blue', 'weight': 2.5, 'fillColor': 'blue', 'fillOpacity': 0.2}, tooltip=folium.GeoJsonTooltip(fields=['cluster'], aliases=['Hotspot ID:'])).add_to(m_hull)
                            st_folium(m_hull, use_container_width=True, height=700)
                        except Exception as e:
                            st.warning(f"Não foi possível desenhar os contornos. Erro: {e}")
                    else:
                        st.warning("Nenhum cluster para desenhar.")
            tab_index += 1

            if df_metas is not None:
                with tabs[tab_index]: # Pacotes de Trabalho
                    cos_simulados = gdf_alocados_final['centro_operativo'].unique() if not gdf_alocados_final.empty else []
                    metas_filtradas = df_metas[df_metas['centro_operativo'].isin(cos_simulados)]
                    
                    st.subheader("Painel de Simulação")
                    if not metas_filtradas.empty:
                        equipes_disponiveis = metas_filtradas['equipes'].sum()
                        meta_diaria_total = metas_filtradas['meta_diária'].sum()
                        metas_filtradas['expectativa_execucao'] = metas_filtradas['equipes'] * metas_filtradas['produção']
                        expectativa_total = metas_filtradas['expectativa_execucao'].sum()
                        
                        servicos_agrupados_para_pacotes = len(gdf_filtrado_base[gdf_filtrado_base['classificacao'] == 'Agrupado'])
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
                        if geometrias_kml_dict:
                            for nome, poligono in geometrias_kml_dict.items():
                                folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 2, 'fillOpacity': 0.1}, tooltip=f"Área de Risco: {nome}").add_to(m_pacotes)
                        if kml_laranja_dict:
                            for nome, poligono in kml_laranja_dict.items():
                                if poligono and not poligono.is_empty:
                                    folium.GeoJson(poligono, style_function=lambda x: {'fillColor': 'orange', 'color': 'orange', 'weight': 2, 'fillOpacity': 0.1}, tooltip=f"Área Laranja: {nome}").add_to(m_pacotes)

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
                            
                            for _, row in gdf_alocados_final.iterrows():
                                folium.CircleMarker(
                                    location=[row['latitude'], row['longitude']],
                                    radius=3,
                                    color=cores_co.get(row['centro_operativo'], 'gray'),
                                    fill=True,
                                    fill_opacity=1,
                                    popup=f"Pacote: {row['pacote_id']}"
                                ).add_to(m_pacotes)
                        
                        st_folium(m_pacotes, use_container_width=True, height=700)
                    else:
                        st.info("Nenhum pacote de trabalho para simular.")
                tab_index += 1
            
            with tabs[tab_index]: # Previsão do Tempo
                st.subheader("Painel de Previsão do Tempo e Contingência")
                api_key = st.secrets.get("OPENWEATHER_API_KEY")

                if not api_key:
                    st.error("Chave da API OpenWeatherMap não encontrada. Por favor, configure-a nos 'secrets' do Streamlit como 'OPENWEATHER_API_KEY'.")
                else:
                    centroids = gdf_filtrado_base.dissolve(by='centro_operativo').centroid
                    co_coords = {co: (point.y, point.x) for co, point in centroids.items()}
                    
                    for co, coords in co_coords.items():
                        with st.expander(f"**{co}**"):
                            forecast_data = get_weather_forecast(coords[0], coords[1], api_key)
                            if isinstance(forecast_data, list):
                                cols = st.columns(len(forecast_data))
                                for i, day in enumerate(forecast_data):
                                    with cols[i]:
                                        st.markdown(f"**{day['date']}**")
                                        st.image(day['icon'], width=60)
                                        st.markdown(day['condition'])
                                        st.markdown(f"Vento: **{day['wind_speed_kmh']} km/h**")
                                        st.markdown(get_operational_status(day['condition'], day['wind_speed_kmh']))
                            else:
                                st.warning(f"Não foi possível obter a previsão para {co}. Erro: {forecast_data}")
            tab_index += 1

            with tabs[tab_index]: # Metodologia
                st.subheader("As Metodologias por Trás da Análise")
                st.markdown("""
                Esta ferramenta utiliza uma combinação de algoritmos geoespaciais e de aprendizado de máquina para fornecer insights sobre a distribuição de serviços.
                - **Detecção de Áreas de Exceção (KML/KMZ):** O script primeiramente lê todos os arquivos `.kml` e `.kmz` da pasta do projeto para identificar polígonos de áreas de risco ou ilhas logísticas. Uma "Área Laranja" de 120 metros é criada ao redor destas áreas como uma zona de pré-risco, que pode ser desativada para áreas específicas na barra lateral. Serviços dentro de ambas as zonas são classificados separadamente e excluídos da análise de clusterização.
                - **Detecção de Hotspots (DBSCAN):** Nos serviços restantes, o DBSCAN é usado para encontrar "hotspots" - áreas de alta concentração de serviços. Ele agrupa pontos densamente próximos e marca como "dispersos" os que estão isolados.
                - **Simulação de Pacotes (Ranking de Densidade):** A lógica para criar pacotes de trabalho prioriza a eficiência. Os hotspots ("Agrupados") são transformados em "pacotes candidatos". Se um hotspot for muito grande para uma única equipe, ele é subdividido de forma inteligente. Todos os candidatos são então ranqueados pela sua densidade (serviços por km²), e os melhores são atribuídos às equipes disponíveis, respeitando o número de **Serviços Designados**.
                """)
                st.subheader("Perguntas Frequentes (FAQ)")
                st.markdown("""
                - **Qual a diferença entre as colunas da planilha de metas?**
                  - **`Produção`**: É a meta de serviços *executados com sucesso* que uma equipe deve atingir. É usada para calcular a "Expectativa de Execução".
                  - **`Serviços Designados`**: É a quantidade total de serviços que devem ser atribuídos a uma equipe para o dia. Este número é geralmente maior que a 'Produção' para compensar a **improdutividade** (ex: cliente ausente). A ferramenta usa os **'Serviços Designados'** para definir o tamanho máximo de um pacote de trabalho.
                  - **`Meta Diária`**: É a meta total do Centro Operativo. É usada para calcular a métrica de "Aderência à Meta".

                - **Qual a estratégia usada para formar os pacotes de trabalho?**
                  - A ferramenta adota uma estratégia de **"Ranking de Densidade"**. Ela primeiro identifica todas as áreas de alta concentração de serviços (hotspots). Em seguida, calcula a densidade de cada uma e cria um ranking. Os pacotes são atribuídos às equipes começando pelos hotspots mais densos, garantindo a máxima eficiência de deslocamento.

                - **O que acontece se um 'hotspot' for muito grande para uma única equipe?**
                  - Se um hotspot contém mais serviços do que o valor em 'Serviços Designados', a ferramenta aplica um método de **"descascamento" (peeling)**: ela "recorta" pacotes de tamanho perfeito de dentro do hotspot, um de cada vez, até que todos os serviços sejam alocados em pacotes que respeitem o limite da equipe.

                - **Por que alguns serviços ficam como "dispersos"?** - Um serviço é considerado disperso se ele não estiver dentro de uma área de risco/laranja e não tiver um número mínimo de vizinhos (`Mínimo de Pontos por Cluster`) dentro de um raio de busca (`Raio do Cluster`).

                - **O que significa um serviço "excedente" na simulação?** - São todos os serviços que não foram alocados em um pacote. Isso inclui os serviços **Dispersos**, os em **Área de Risco**, os em **Área Laranja**, e os **Agrupados** que não entraram no ranking dos melhores pacotes (seja por baixa densidade ou por falta de equipes disponíveis).
                """)
        else:
            st.warning("Nenhum dado para exibir com os filtros atuais.")
else:
    st.info("Aguardando o upload de um arquivo para iniciar a análise.")
